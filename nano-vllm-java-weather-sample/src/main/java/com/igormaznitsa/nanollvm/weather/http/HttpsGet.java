package com.igormaznitsa.nanollvm.weather.http;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * One JSON GET. The body is capped so a redirect cannot fill the heap.
 */
public final class HttpsGet {

  private static final int MAX_CHARS = 1_000_000;
  private static final HttpClient CLIENT = HttpClient.newBuilder()
    .connectTimeout(Duration.ofSeconds(5))
    .followRedirects(HttpClient.Redirect.NORMAL)
    .build();

  private HttpsGet() {
  }

  public static String body(final URI uri) throws IOException, InterruptedException {
    requireNonNull(uri, "uri");
    HttpRequest request = HttpRequest.newBuilder(uri)
      .timeout(Duration.ofSeconds(12))
      .header("Accept", "application/json")
      .header("User-Agent", "nano-vllm-java-weather-sample")
      .GET()
      .build();
    HttpResponse<String> response = CLIENT.send(request, HttpResponse.BodyHandlers.ofString(UTF_8));
    if (response.statusCode() != 200) {
      throw new IOException("HTTP " + response.statusCode());
    }
    String payload = response.body();
    if (payload == null) {
      throw new IOException("empty response");
    }
    if (payload.length() > MAX_CHARS) {
      throw new IOException("response is too large");
    }
    return payload;
  }
}
