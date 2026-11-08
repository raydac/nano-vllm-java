package com.igormaznitsa.nanollvm.weather;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import java.nio.file.Path;

/**
 * Asks a tool-capable chat model for this computer's city, then for the weather there,
 * prints the temperature and sky, and speaks that line with Piper. Before each lookup
 * Piper says what it is doing: looking for the location, then getting the weather in
 * the city that lookup found. Each line is heard to the end before the next step starts.
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
    } catch (WeatherModels.MissingModels ex) {
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
      Forecast forecast = ask(chat, voice, conditions, timer);
      speak(voice, forecast, timer);
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

  private static Forecast ask(
    final LLM chat,
    final LLM voice,
    final LocalConditions conditions,
    final StepTimer timer
  ) {
    ChatSession session = chat.chat()
      .enableThinking(false)
      .maxTokens(256)
      .deterministic();
    say(voice, null, "start", "I am a weather station.");

    say(voice, timer, "location", lookingForLocation());

    timer.measure("place turn", () -> System.out.println(session.send(ASK_PLACE).answer()));
    conditions.place().ifPresent(place ->
      say(voice, timer, "weather", gettingWeather(place)));
    timer.measure("weather turn", () -> System.out.println(session.send(ASK_WEATHER).answer()));
    return conditions.forecast().orElseThrow(() -> new IllegalStateException(
      "the model did not call current_location and current_weather"));
  }

  private static void speak(
    final LLM voice,
    final Forecast forecast,
    final StepTimer timer
  ) {
    String speech = forecast.speech();
    System.out.println();
    System.out.println(forecast.logLine());
    System.out.println("Weather data by Open-Meteo.com (CC BY 4.0).");
    System.out.println(speech);
    play(synthesize(voice, timer, "synthesize", speech));
  }

  private static void say(
    final LLM voice,
    final StepTimer timer,
    final String stage,
    final String line
  ) {
    System.out.println();
    System.out.println(line);
    play(synthesize(voice, timer, "synthesize " + stage, line));
  }

  static String lookingForLocation() {
    return "I am looking for location.";
  }

  static String gettingWeather(final Place place) {
    requireNonNull(place, "place");
    String city = place.city().isEmpty() ? place.where() : place.city();
    return "I am getting weather info in %s.".formatted(city);
  }

  private static byte[] synthesize(
    final LLM voice,
    final StepTimer timer,
    final String step,
    final String text
  ) {
    final LlmOutSoundData sound;
    if (timer == null) {
      sound = voice.generate(LlmInText.of(text), LlmModality.AUDIO);
    } else {
      sound = timer.measure(
        step,
        () -> voice.generate(LlmInText.of(text), LlmModality.AUDIO));
    }
    return sound.wav();
  }

  private static void play(final byte[] wav) {
    new SoundClip(wav).play();
  }

}
