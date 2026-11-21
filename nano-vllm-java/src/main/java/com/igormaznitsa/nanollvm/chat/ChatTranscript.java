package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Session transcript plus the filters that admit each message into history, the model prompt, or
 * both.
 */
final class ChatTranscript {

  private final List<ChatMessageFilter> filters = new ArrayList<>();
  private final List<Entry> entries = new ArrayList<>();

  void addFilter(final ChatMessageFilter filter) {
    this.filters.add(requireNonNull(filter, "filter"));
  }

  void addUnfiltered(final List<ChatMessage> messages) {
    requireNonNull(messages, "messages");
    for (ChatMessage message : messages) {
      this.entries.add(new Entry(requireNonNull(message, "message"), true));
    }
  }

  void clear() {
    this.entries.clear();
  }

  /**
   * Re-admits every stored message. A message both flags reject is removed. Returns {@code true}
   * when at least one message was removed.
   */
  boolean reapply() {
    int before = this.entries.size();
    List<Entry> kept = new ArrayList<>(this.entries.size());
    for (Entry entry : this.entries) {
      ChatAdmission admission = this.admit(entry.message());
      if (!admission.keepInHistory() && !admission.sendToModel()) {
        continue;
      }
      kept.add(new Entry(entry.message(), admission.keepInHistory()));
    }
    this.entries.clear();
    this.entries.addAll(kept);
    return this.entries.size() != before;
  }

  /**
   * Appends {@code message} when at least one destination is open.
   *
   * @return the stored entry, or {@code null} when both destinations are closed
   */
  Entry add(final ChatMessage message) {
    requireNonNull(message, "message");
    ChatAdmission admission = this.admit(message);
    if (!admission.keepInHistory() && !admission.sendToModel()) {
      return null;
    }
    Entry entry = new Entry(message, admission.keepInHistory());
    this.entries.add(entry);
    return entry;
  }

  ChatAdmission admit(final ChatMessage message) {
    requireNonNull(message, "message");
    ChatAdmission admission = ChatAdmission.both();
    for (ChatMessageFilter filter : this.filters) {
      admission = admission.and(requireNonNull(filter.admit(message), "admission"));
    }
    return admission;
  }

  List<ChatMessage> history() {
    return this.entries.stream()
      .filter(Entry::inHistory)
      .map(Entry::message)
      .toList();
  }

  /**
   * Messages whose model flag is open. When {@code current} is still stored and its text differs
   * from {@code modelText}, the prompt uses {@code modelText} for that one entry.
   */
  List<ChatMessage> modelMessages(final Entry current, final String modelText) {
    requireNonNull(modelText, "modelText");
    List<ChatMessage> visible = new ArrayList<>();
    for (Entry entry : this.entries) {
      if (!this.admit(entry.message()).sendToModel()) {
        continue;
      }
      if (entry == current && !entry.message().content().equals(modelText)) {
        visible.add(ChatMessage.user(modelText));
      } else {
        visible.add(entry.message());
      }
    }
    return List.copyOf(visible);
  }

  boolean contains(final Entry entry) {
    return this.entries.stream().anyMatch(candidate -> candidate == entry);
  }

  int size() {
    return this.entries.size();
  }

  ChatRole firstRole() {
    return this.entries.isEmpty() ? null : this.entries.getFirst().message().role();
  }

  void scrub(final Predicate<String> match, final String replacement) {
    requireNonNull(match, "match");
    String text = requireNonNull(replacement, "replacement");
    for (int i = 0; i < this.entries.size(); i++) {
      Entry entry = this.entries.get(i);
      if (entry.message().role() == ChatRole.ASSISTANT && match.test(entry.message().content())) {
        this.entries.set(i, new Entry(ChatMessage.assistant(text), entry.inHistory()));
      }
    }
  }

  void trimHistory(final int max) {
    while (this.historyCount() > max) {
      int dropAt = this.oldestHistoryIndex();
      if (dropAt < 0) {
        return;
      }
      this.entries.remove(dropAt);
    }
  }

  void dropOldest() {
    int dropAt = this.firstRole() == ChatRole.SYSTEM ? 1 : 0;
    if (dropAt >= this.entries.size()) {
      return;
    }
    this.entries.remove(dropAt);
    if (dropAt < this.entries.size()
      && this.entries.get(dropAt).message().role() == ChatRole.ASSISTANT) {
      this.entries.remove(dropAt);
    }
  }

  private int historyCount() {
    return (int) this.entries.stream().filter(Entry::inHistory).count();
  }

  private int oldestHistoryIndex() {
    int start = this.firstRole() == ChatRole.SYSTEM && this.entries.getFirst().inHistory() ? 1 : 0;
    for (int i = start; i < this.entries.size(); i++) {
      if (this.entries.get(i).inHistory()) {
        return i;
      }
    }
    return -1;
  }

  static final class Entry {
    private final ChatMessage message;
    private final boolean inHistory;

    private Entry(final ChatMessage message, final boolean inHistory) {
      this.message = message;
      this.inHistory = inHistory;
    }

    ChatMessage message() {
      return this.message;
    }

    boolean inHistory() {
      return this.inHistory;
    }
  }
}
