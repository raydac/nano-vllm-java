/**
 * Non-exported runnable demos ({@link Example}, {@link Bench}, {@link LogTriageHelloWorld}).
 * Helpers for local {@code models/} and {@code rag/} live in {@code samples.utils}.
 * Not wired as {@code Main-Class} in the published library JAR — launch from the repo with
 * {@code mvn exec:java} (default main {@link Example}) or
 * {@code java -m com.igormaznitsa.nanollvm/com.igormaznitsa.nanollvm.samples.Example}.
 * Library consumers should depend on {@code models}, {@code llm}, {@code chat}, {@code rag}, etc.
 */
package com.igormaznitsa.nanollvm.samples;
