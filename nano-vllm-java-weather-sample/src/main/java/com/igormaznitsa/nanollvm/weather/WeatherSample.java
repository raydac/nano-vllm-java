package com.igormaznitsa.nanollvm.weather;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.weather.forecast.Forecast;
import com.igormaznitsa.nanollvm.weather.model.MissingModels;
import com.igormaznitsa.nanollvm.weather.model.WeatherModels;
import com.igormaznitsa.nanollvm.weather.speech.WavSpeaker;
import com.igormaznitsa.nanollvm.weather.time.StepTimer;
import com.igormaznitsa.nanollvm.weather.tools.LocalConditions;
import java.nio.file.Path;

/**
 * Asks a tool-capable chat model for this computer's city, then for the weather there,
 * prints the temperature and sky, and speaks that line with Piper.
 *
 * <p>Default chat checkpoint is Qwen3-0.6B. One question that names both tools makes that
 * model call only one of them, so the place and the weather are two turns. Gemma 3 cannot
 * take these tools. An optional first argument selects another tool-capable checkpoint
 * (Qwen3.5, LFM2, or Gemma 4).
 *
 * <p>From the repository root, after {@code models/download-qwen3-0.6b.sh} and a Piper voice:
 * {@code mvn -pl nano-vllm-java-weather-sample -am -q exec:java}
 */
public final class WeatherSample {

  private static final String SYSTEM = """
    Call the tool named in the question before you answer.
    Do not invent a city, a temperature, or the weather.""";

  private static final String ASK_PLACE =
    "Find the current location. What city is this computer in? Reply with the place only.";

  private static final String ASK_WEATHER =
    "Find out the current weather in the location. What is the temperature and the weather there? Reply with the temperature and the weather.";

  private WeatherSample() {
  }

  public static void main(final String[] args) {
    WeatherModels.Required models;
    try {
      models = WeatherModels.require(args);
    } catch (MissingModels ex) {
      System.err.println(ex.getMessage());
      System.exit(1);
      return;
    }
    Path chatDir = models.chat();
    Path voiceDir = models.voice();
    announce(chatDir, voiceDir);

    StepTimer timer = new StepTimer();
    LocalConditions conditions = new LocalConditions(timer);
    try (LlmModel chatModel = timer.measure("load chat model", () -> LlmModelFactory.make(chatDir));
         LlmModel voiceModel = timer.measure("load voice", () -> openVoice(voiceDir));
         LLM chat = timer.measure("build chat engine", () -> chatEngine(chatModel, conditions));
         LLM voice = timer.measure("build voice engine", () -> LLM.builder(voiceModel).build())) {
      speak(voice, voiceDir, ask(chat, conditions, timer), timer);
    } finally {
      timer.logTotal();
    }
  }

  private static void announce(final Path chatDir, final Path voiceDir) {
    System.out.println("chat=" + chatDir);
    System.out.println("voice=" + voiceDir);
  }

  private static LlmModel openVoice(final Path voiceDir) {
    return LlmModelFactory.open(voiceDir)
      .optionalData(LlmOptionalData.ESPEAK_DATA, voiceDir.resolve("espeak-ng-data"))
      .make();
  }

  private static LLM chatEngine(final LlmModel model, final LocalConditions conditions) {
    return LLM.builder(model)
      .systemPrompt(SYSTEM.strip())
      .tool(conditions.locationTool(), conditions.weatherTool())
      .build();
  }

  private static Forecast ask(final LLM chat, final LocalConditions conditions,
                              final StepTimer timer) {
    ChatSession session = chat.chat()
      .enableThinking(false)
      .maxTokens(256)
      .deterministic();
    System.out.println();
    timer.measure("place turn", () -> System.out.println(session.send(ASK_PLACE).answer()));
    System.out.println();
    timer.measure("weather turn", () -> System.out.println(session.send(ASK_WEATHER).answer()));
    return conditions.forecast().orElseThrow(() -> new IllegalStateException(
      "the model did not call current_location and current_weather"));
  }

  private static void speak(
    final LLM voice,
    final Path voiceDir,
    final Forecast forecast,
    final StepTimer timer
  ) {
    String speech = forecast.speech(WeatherModels.englishVoice(voiceDir));
    System.out.println();
    System.out.println(forecast.logLine());
    System.out.println("Weather data by Open-Meteo.com (CC BY 4.0).");
    System.out.println(speech);
    LlmOutSoundData sound = timer.measure(
      "synthesize",
      () -> voice.generate(LlmInText.of(speech), LlmModality.AUDIO));
    boolean played = timer.measure("play", () -> WavSpeaker.play(sound.wav()));
    System.out.println(played
      ? "Played the forecast."
      : "No default audio line; the forecast was not played.");
  }
}
