package com.igormaznitsa.nanollvm.weather;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.ModelTool;
import java.util.Optional;

/**
 * Place and forecast filled by the chat tools. Both tools ignore arguments, so a call
 * cannot choose its own URL. The chat thread is the only writer.
 */
public final class LocalConditions {

  private static final String EMPTY_SCHEMA = "{\"type\":\"object\",\"properties\":{}}";

  private final StepTimer timer;
  private Place place;
  private Forecast forecast;

  public LocalConditions(final StepTimer timer) {
    this.timer = requireNonNull(timer, "timer");
  }

  private static String shown(final String arguments) {
    if (arguments == null || arguments.isBlank()) {
      return "";
    }
    String oneLine = arguments.replaceAll("\\s+", " ").strip();
    return oneLine.length() > 200 ? oneLine.substring(0, 200) : oneLine;
  }

  public ModelTool locationTool() {
    return ModelTool.of(
      "current_location",
      "City and country of this computer. No arguments.",
      EMPTY_SCHEMA,
      this::reportLocation);
  }

  public ModelTool weatherTool() {
    return ModelTool.of(
      "current_weather",
      "Temperature and weather at that city. Call current_location first. No arguments.",
      EMPTY_SCHEMA,
      this::reportWeather);
  }

  public Optional<Place> place() {
    return Optional.ofNullable(this.place);
  }

  public Optional<Forecast> forecast() {
    return Optional.ofNullable(this.forecast);
  }

  private String reportLocation(final String arguments) {
    System.err.println("Requested location: " + shown(arguments));

    long began = System.nanoTime();
    Optional<Place> found = this.locate();
    long elapsed = System.nanoTime() - began;

    String value = found.map(Place::forModel)
      .orElse("error: could not resolve this computer's location");
    System.err.println("[tool] current_location");
    System.err.println(value);
    System.err.flush();
    found.ifPresent(place -> System.out.println(WorldCard.of(place)));
    this.timer.log("location lookup", elapsed);
    return value;
  }

  private String reportWeather(final String arguments) {
    System.err.println("Requested weather: " + shown(arguments));

    long began = System.nanoTime();
    String value = this.weather();
    long elapsed = System.nanoTime() - began;

    System.err.println("[tool] current_weather");
    System.err.println(value);
    if (this.place != null) {
      this.timer.log("weather lookup", elapsed);
    }
    return value;
  }

  private Optional<Place> locate() {
    try {
      this.place = IpPlace.locate();
      this.forecast = null;
      return Optional.of(this.place);
    } catch (IllegalStateException ex) {
      return Optional.empty();
    }
  }

  private String weather() {
    if (this.place == null) {
      return "error: call current_location first";
    }
    try {
      this.forecast = Forecast.at(this.place);
      return this.forecast.logLine();
    } catch (IllegalStateException ex) {
      return "error: could not read the weather";
    }
  }
}
