package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;

import com.googlecode.lanterna.TerminalSize;
import com.googlecode.lanterna.bundle.LanternaThemes;
import com.googlecode.lanterna.gui2.BasicWindow;
import com.googlecode.lanterna.gui2.BorderLayout;
import com.googlecode.lanterna.gui2.Borders;
import com.googlecode.lanterna.gui2.Button;
import com.googlecode.lanterna.gui2.Component;
import com.googlecode.lanterna.gui2.Direction;
import com.googlecode.lanterna.gui2.EmptySpace;
import com.googlecode.lanterna.gui2.Label;
import com.googlecode.lanterna.gui2.LinearLayout;
import com.googlecode.lanterna.gui2.MultiWindowTextGUI;
import com.googlecode.lanterna.gui2.Panel;
import com.googlecode.lanterna.gui2.RadioBoxList;
import com.googlecode.lanterna.gui2.TextBox;
import com.googlecode.lanterna.gui2.Window;
import com.googlecode.lanterna.gui2.WindowListenerAdapter;
import com.googlecode.lanterna.gui2.dialogs.MessageDialog;
import com.googlecode.lanterna.gui2.dialogs.MessageDialogButton;
import com.googlecode.lanterna.input.KeyDecodingProfile;
import com.googlecode.lanterna.input.KeyStroke;
import com.googlecode.lanterna.input.KeyType;
import com.googlecode.lanterna.screen.Screen;
import com.googlecode.lanterna.screen.TerminalScreen;
import com.googlecode.lanterna.terminal.DefaultTerminalFactory;
import com.googlecode.lanterna.terminal.ExtendedTerminal;
import com.googlecode.lanterna.terminal.MouseCaptureMode;
import com.googlecode.lanterna.terminal.Terminal;
import com.googlecode.lanterna.terminal.ansi.StreamBasedTerminal;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmTextKind;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.CausalBundle;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.ModelChoice;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.RagMode;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.RagSetup;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

final class ExampleTui {

  private ExampleTui() {
  }

  static void run(final String[] args) throws Exception {
    DefaultTerminalFactory factory = new DefaultTerminalFactory()
      .setTerminalEmulatorTitle("nano-vllm Example")
      .setForceTextTerminal(true)
      .setMouseCaptureMode(MouseCaptureMode.CLICK_RELEASE);
    Terminal terminal = factory.createTerminal();
    Screen screen = new TerminalScreen(terminal);
    screen.startScreen();
    enableMouseCapture(terminal);

    MultiWindowTextGUI gui = new MultiWindowTextGUI(screen);
    gui.setTheme(LanternaThemes.getDefaultTheme());
    ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
      Thread t = new Thread(r, "example-tui-worker");
      t.setDaemon(true);
      return t;
    });

    try {
      Optional<Path> explicit = resolveExplicitPath(args);
      SetupResult setup = showSetup(gui, explicit);
      if (setup == null) {
        return;
      }
      LoadedSession loaded = loadSession(gui, worker, setup);
      if (loaded == null) {
        return;
      }
      try (loaded) {
        if (loaded.model().isEmbeddingModel()) {
          showEmbed(gui, worker, loaded);
        } else {
          showChat(gui, worker, loaded);
        }
      }
    } finally {
      worker.shutdownNow();
      disableMouseCapture(terminal);
      screen.stopScreen();
    }
  }

  private static void enableMouseCapture(final Terminal terminal) {
    if (terminal instanceof StreamBasedTerminal stream) {
      stream.getInputDecoder().addProfile(sgrMouseProfile());
    }
    if (terminal instanceof ExtendedTerminal extended) {
      try {
        // CLICK_RELEASE enables xterm 1000; most modern terminals then emit SGR if 1006 is also on.
        extended.setMouseCaptureMode(MouseCaptureMode.CLICK_RELEASE);
      } catch (Exception ignored) {
        // Terminal rejected mouse capture; keyboard navigation still works.
      }
    }
    writeTerminal(terminal, "\u001b[?1000h\u001b[?1006h");
  }

  private static void disableMouseCapture(final Terminal terminal) {
    writeTerminal(terminal, "\u001b[?1006l\u001b[?1000l");
    if (terminal instanceof ExtendedTerminal extended) {
      try {
        extended.setMouseCaptureMode(null);
      } catch (Exception ignored) {
        // Best-effort restore when leaving the TUI.
      }
    }
  }

  private static KeyDecodingProfile sgrMouseProfile() {
    return () -> List.of(new SgrMouseCharacterPattern());
  }

  private static void writeTerminal(final Terminal terminal, final String sequence) {
    try {
      for (int i = 0; i < sequence.length(); i++) {
        terminal.putCharacter(sequence.charAt(i));
      }
      terminal.flush();
    } catch (Exception ignored) {
      // Best-effort mouse mode toggle.
    }
  }

  private static Optional<Path> resolveExplicitPath(final String[] args) {
    boolean explicit = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      || NanoLlvmProps.systemProperty(PROP_MODEL) != null
      || NanoLlvmProps.environment(ENV_MODEL) != null;
    if (!explicit) {
      return Optional.empty();
    }
    return Optional.of(BundledModels.resolveDefault(args));
  }

  private static SetupResult showSetup(
    final MultiWindowTextGUI gui,
    final Optional<Path> explicitPath
  ) {
    List<ModelChoice> catalog = ExampleSessionSupport.catalog();
    List<ModelChoice> availableModels = catalog.stream()
      .filter(choice -> !choice.missing())
      .toList();
    List<ModelChoice> missingModels = catalog.stream()
      .filter(ModelChoice::missing)
      .toList();

    RadioBoxList<ModelChoice> modelList = new RadioBoxList<>();
    for (ModelChoice choice : availableModels) {
      modelList.addItem(choice);
    }
    if (!availableModels.isEmpty()) {
      modelList.setCheckedItemIndex(0);
      modelList.setSelectedIndex(0);
    }

    boolean gtePresent = BundledModels.find(BundledModels.GTE_SMALL_GGUF).isPresent();
    boolean ragCorpus = BundledRag.find().isPresent();
    List<RagChoice> availableRag = ragChoices(gtePresent).stream()
      .filter(RagChoice::enabled)
      .toList();
    List<RagChoice> missingRag = ragChoices(gtePresent).stream()
      .filter(choice -> !choice.enabled())
      .toList();

    RadioBoxList<RagChoice> ragList = new RadioBoxList<>();
    for (RagChoice choice : availableRag) {
      ragList.addItem(choice);
    }
    if (!availableRag.isEmpty()) {
      ragList.setCheckedItemIndex(0);
      ragList.setSelectedIndex(0);
    }
    if (!ragCorpus) {
      ragList.setEnabled(false);
    }

    Label hint = new Label(explicitPath
      .map(path -> "Model path: " + path)
      .orElse("Pick a downloaded model, then RAG mode, then Load.  (mouse click supported)"));
    if (!ragCorpus) {
      hint.setText(hint.getText() + "  (no rag/ corpus — plain chat only)");
    }
    if (availableModels.isEmpty() && explicitPath.isEmpty()) {
      hint.setText(
        hint.getText() + "  (no models downloaded — use download scripts under models/)");
    }

    AtomicBoolean cancelled = new AtomicBoolean(true);
    BasicWindow window = new BasicWindow("nano-vllm Example");
    window.setHints(Set.of(Window.Hint.CENTERED));

    Button load = actionButton("Load", () -> {
      cancelled.set(false);
      window.close();
    });
    load.setEnabled(explicitPath.isPresent() || !availableModels.isEmpty());

    Panel root = new Panel(new LinearLayout(Direction.VERTICAL));
    root.addComponent(hint);
    root.addComponent(new EmptySpace(TerminalSize.ONE));

    if (explicitPath.isEmpty()) {
      Panel modelPanel = new Panel(new LinearLayout(Direction.VERTICAL));
      if (availableModels.isEmpty()) {
        modelPanel.addComponent(new Label("(none available)"));
      } else {
        modelPanel.addComponent(modelList);
      }
      for (ModelChoice missing : missingModels) {
        modelPanel.addComponent(new Label("  · " + missing.label() + "  [not downloaded]"));
      }
      root.addComponent(modelPanel.withBorder(Borders.singleLine("Model")));
    }

    Panel ragPanel = new Panel(new LinearLayout(Direction.VERTICAL));
    if (!ragCorpus) {
      ragPanel.addComponent(new Label("(no rag/ corpus)"));
    } else {
      ragPanel.addComponent(ragList);
      for (RagChoice missing : missingRag) {
        ragPanel.addComponent(new Label("  · " + missing.mode().label() + "  [not downloaded]"));
      }
    }
    root.addComponent(ragPanel.withBorder(Borders.singleLine("RAG")));
    root.addComponent(new EmptySpace(TerminalSize.ONE));
    root.addComponent(
      centeredButtons(load, actionButton("Quit", () -> {
        cancelled.set(true);
        window.close();
      })),
      LinearLayout.createLayoutData(LinearLayout.Alignment.Fill));

    window.setComponent(root);
    if (explicitPath.isEmpty() && !availableModels.isEmpty()) {
      modelList.takeFocus();
    } else if (ragCorpus) {
      ragList.takeFocus();
    }
    gui.addWindowAndWait(window);

    if (cancelled.get()) {
      return null;
    }

    Path modelPath;
    try {
      modelPath = explicitPath.orElseGet(() -> {
        ModelChoice choice = modelList.getCheckedItem();
        if (choice == null || choice.missing()) {
          throw new IllegalStateException("Select a downloaded model.");
        }
        return choice.requirePath();
      });
    } catch (RuntimeException ex) {
      MessageDialog.showMessageDialog(gui, "Model", ex.getMessage(), MessageDialogButton.OK);
      return showSetup(gui, explicitPath);
    }

    RagMode ragMode = RagMode.NONE;
    if (ragCorpus) {
      RagChoice choice = ragList.getCheckedItem();
      ragMode = choice == null || !choice.enabled() ? RagMode.NONE : choice.mode();
    }

    return new SetupResult(modelPath, ragMode);
  }

  private static LoadedSession loadSession(
    final MultiWindowTextGUI gui,
    final ExecutorService worker,
    final SetupResult setup
  ) throws Exception {
    BasicWindow progress = new BasicWindow("Loading");
    progress.setHints(Set.of(Window.Hint.CENTERED, Window.Hint.MODAL));
    Label status = new Label("Loading model from " + setup.modelPath());
    status.setPreferredSize(new TerminalSize(72, 4));
    progress.setComponent(status.withBorder(Borders.singleLine("Status")));
    gui.addWindow(progress);

    LlmListener listener = (source, event) -> {
      if (event.kind() == LlmTextKind.STATUS_INFO || event.kind() == LlmTextKind.STATUS_PROGRESS) {
        String text = event.text() == null ? "" : event.text().strip();
        if (!text.isEmpty()) {
          gui.getGUIThread().invokeLater(() -> status.setText(truncate(text, 280)));
        }
      }
    };

    AtomicBoolean failed = new AtomicBoolean(false);
    String[] error = {null};
    LoadedSession[] result = {null};

    worker.submit(() -> {
      try {
        Path path = setup.modelPath();
        LlmModel model = LlmModelFactory.make(path, listener);
        RagSetup ragSetup = new RagSetup(null, null);
        if (!model.isEmbeddingModel() && setup.ragMode() != RagMode.NONE) {
          Optional<Path> ragRoot = BundledRag.find();
          Optional<Path> gte = BundledModels.find(BundledModels.GTE_SMALL_GGUF);
          if (ragRoot.isPresent()) {
            RagSetup prepared = ExampleSessionSupport.prepareRagSetup(
              ragRoot.get(),
              setup.ragMode(),
              gte.orElse(null),
              ExampleSessionSupport.isGemmaPath(path),
              listener,
              msg -> gui.getGUIThread().invokeLater(() -> status.setText(truncate(msg, 280))));
            if (prepared != null) {
              ragSetup = prepared;
            }
          }
        }

        CausalBundle causal = null;
        if (!model.isEmbeddingModel()) {
          causal = ExampleSessionSupport.openCausal(
            model,
            ragSetup,
            listener,
            msg -> gui.getGUIThread().invokeLater(() -> status.setText(truncate(msg, 280))));
        }
        result[0] = new LoadedSession(model, ragSetup, causal);
      } catch (Throwable t) {
        failed.set(true);
        error[0] = t.getMessage() == null ? t.toString() : t.getMessage();
      } finally {
        gui.getGUIThread().invokeLater(progress::close);
      }
    });

    gui.waitForWindowToClose(progress);

    if (failed.get()) {
      MessageDialog.showMessageDialog(
        gui,
        "Load failed",
        error[0] == null ? "Unknown error" : error[0],
        MessageDialogButton.OK);
      return null;
    }
    return result[0];
  }

  private static void showChat(
    final MultiWindowTextGUI gui,
    final ExecutorService worker,
    final LoadedSession loaded
  ) {
    CausalBundle causal = loaded.causal();
    SoftWrappedPane thinking = softPane(5);
    SoftWrappedPane answer = softPane(12);
    TextBox input = editablePane(3);
    Label status = new Label("Ready — type below, then Send (or Ctrl+Enter)");
    AtomicBoolean busy = new AtomicBoolean(false);

    LlmListener stream = (source, event) -> gui.getGUIThread().invokeLater(() -> {
      switch (event.kind()) {
        case TEXT_THINKING -> thinking.append(event.text());
        case TEXT_ASSISTANT -> answer.append(event.text());
        case TEXT_ADVISOR_NOTE -> thinking.append(
          "[" + event.advisorName() + "] " + event.text() + "\n");
        case STATUS_INFO, STATUS_PROGRESS -> {
          String text = event.text() == null ? "" : event.text().strip();
          if (!text.isEmpty()) {
            status.setText(truncate(text, 100));
          }
        }
        default -> {
        }
      }
    });
    if (causal.rag() != null) {
      causal.rag().listen(stream);
    } else {
      causal.chat().listen(stream);
    }

    BasicWindow window = new BasicWindow(
      causal.ragMode() ? "Chat + RAG" : "Chat — " + loaded.model().architectureName());
    window.setHints(Set.of(Window.Hint.FULL_SCREEN, Window.Hint.NO_DECORATIONS));

    Runnable clearHistory = () -> {
      if (busy.get()) {
        return;
      }
      if (causal.rag() != null) {
        causal.rag().clear();
      } else {
        causal.chat().clear();
      }
      thinking.clear();
      answer.clear();
      status.setText("Conversation cleared");
    };

    Runnable send = () -> {
      if (!busy.compareAndSet(false, true)) {
        return;
      }
      String user = input.getText().strip();
      if (user.isEmpty()) {
        busy.set(false);
        return;
      }
      String command = user.toLowerCase(Locale.ROOT);
      if (command.equals("/exit") || command.equals("/quit")
        || command.equals("exit") || command.equals("quit")) {
        window.close();
        return;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        clearHistory.run();
        input.setText("");
        busy.set(false);
        return;
      }

      answer.appendTurn(user);
      thinking.clear();
      input.setText("");
      status.setText("Generating…");
      input.setEnabled(false);

      worker.submit(() -> {
        try {
          ChatReply reply = causal.rag() != null
            ? causal.rag().send(user)
            : causal.chat().send(user);
          int tokens = reply.stats().completionTokens();
          double sec = Math.max(1L, reply.stats().elapsedNanos()) / 1e9;
          StringBuilder meta = new StringBuilder(String.format(
            Locale.ROOT,
            "%d tok in %.2fs → %.1f tok/s",
            tokens,
            sec,
            reply.stats().completionTokensPerSecond()));
          if (causal.rag() != null) {
            List<RagHit> hits = causal.rag().lastHits();
            if (hits.isEmpty()) {
              meta.append(" · no RAG hits");
            } else {
              String sources = hits.stream()
                .map(hit -> Path.of(hit.chunk().source()).getFileName().toString())
                .distinct()
                .collect(Collectors.joining(", "));
              meta.append(" · ").append(hits.size()).append(" chunk(s): ").append(sources);
            }
          }
          gui.getGUIThread().invokeLater(() -> status.setText(meta.toString()));
        } catch (Throwable t) {
          gui.getGUIThread().invokeLater(() -> {
            answer.append("\n[error] " + t.getMessage() + "\n");
            status.setText("Error");
          });
        } finally {
          gui.getGUIThread().invokeLater(() -> {
            input.setEnabled(true);
            busy.set(false);
            input.takeFocus();
          });
        }
      });
    };

    Panel root = new Panel(new BorderLayout());
    root.addComponent(status.withBorder(Borders.singleLine("Status")), BorderLayout.Location.TOP);

    Panel center = new Panel(new LinearLayout(Direction.VERTICAL));
    center.addComponent(
      thinking.component().withBorder(Borders.singleLine("Thinking / advisors")),
      LinearLayout.createLayoutData(LinearLayout.Alignment.Fill, LinearLayout.GrowPolicy.CanGrow));
    center.addComponent(
      answer.component().withBorder(Borders.singleLine("Answer")),
      LinearLayout.createLayoutData(LinearLayout.Alignment.Fill, LinearLayout.GrowPolicy.CanGrow));
    root.addComponent(center, BorderLayout.Location.CENTER);

    root.addComponent(
      composePanel(
        input,
        actionButton("Send", send),
        actionButton("Clear", clearHistory),
        actionButton("Quit", window::close),
        "Ctrl+Enter = Send · Tab moves to buttons · F5 = Clear · Esc = Quit"),
      BorderLayout.Location.BOTTOM);

    window.setComponent(root);
    window.addWindowListener(new WindowListenerAdapter() {
      @Override
      public void onResized(
        final Window basePane,
        final TerminalSize oldSize,
        final TerminalSize newSize
      ) {
        thinking.refresh();
        answer.refresh();
        reflowEditable(input);
      }

      @Override
      public void onUnhandledInput(
        final Window basePane,
        final KeyStroke keyStroke,
        final AtomicBoolean deliverEvent
      ) {
        if (keyStroke.getKeyType() == KeyType.Escape) {
          window.close();
          deliverEvent.set(true);
        } else if (keyStroke.getKeyType() == KeyType.F5) {
          clearHistory.run();
          deliverEvent.set(true);
        } else if (keyStroke.getKeyType() == KeyType.Enter && keyStroke.isCtrlDown()) {
          send.run();
          deliverEvent.set(true);
        }
      }
    });

    input.takeFocus();
    gui.addWindowAndWait(window);
  }

  private static void showEmbed(
    final MultiWindowTextGUI gui,
    final ExecutorService worker,
    final LoadedSession loaded
  ) {
    SoftWrappedPane result = softPane(14);
    TextBox input = editablePane(3);
    Label status = new Label("Ready — type text below, then Embed (or Ctrl+Enter)");
    AtomicBoolean busy = new AtomicBoolean(false);
    float[][] previous = {null};

    BasicWindow window = new BasicWindow("Embeddings — " + loaded.model().architectureName());
    window.setHints(Set.of(Window.Hint.FULL_SCREEN, Window.Hint.NO_DECORATIONS));

    Runnable clear = () -> {
      previous[0] = null;
      result.clear();
      status.setText("Previous embedding cleared");
    };

    Runnable embed = () -> {
      if (!busy.compareAndSet(false, true)) {
        return;
      }
      String user = input.getText().strip();
      if (user.isEmpty()) {
        busy.set(false);
        return;
      }
      String command = user.toLowerCase(Locale.ROOT);
      if (command.equals("/exit") || command.equals("/quit")
        || command.equals("exit") || command.equals("quit")) {
        window.close();
        return;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        clear.run();
        input.setText("");
        busy.set(false);
        return;
      }

      result.append("text> " + user + "\n");
      input.setText("");
      status.setText("Embedding…");
      input.setEnabled(false);

      worker.submit(() -> {
        try {
          long started = System.nanoTime();
          float[] vector = loaded.model().embed(user);
          double elapsedSec = (System.nanoTime() - started) / 1e9;
          String line = String.format(
            Locale.ROOT,
            "dim=%d  L2=%.4f  %.3fs  preview=%s%n",
            vector.length,
            ExampleSessionSupport.l2Norm(vector),
            elapsedSec,
            ExampleSessionSupport.preview(vector));
          if (previous[0] != null) {
            line += String.format(
              Locale.ROOT,
              "cos(prev)=%.4f%n",
              ExampleSessionSupport.cosine(previous[0], vector));
          }
          previous[0] = vector;
          String out = line;
          gui.getGUIThread().invokeLater(() -> {
            result.append(out + "\n");
            status.setText(String.format(Locale.ROOT, "dim=%d · %.3fs", vector.length, elapsedSec));
          });
        } catch (Throwable t) {
          gui.getGUIThread().invokeLater(() -> {
            result.append("[error] " + t.getMessage() + "\n");
            status.setText("Error");
          });
        } finally {
          gui.getGUIThread().invokeLater(() -> {
            input.setEnabled(true);
            busy.set(false);
            input.takeFocus();
          });
        }
      });
    };

    Panel root = new Panel(new BorderLayout());
    root.addComponent(status.withBorder(Borders.singleLine("Status")), BorderLayout.Location.TOP);
    root.addComponent(
      result.component().withBorder(Borders.singleLine("Result")),
      BorderLayout.Location.CENTER);
    root.addComponent(
      composePanel(
        input,
        actionButton("Embed", embed),
        actionButton("Clear", clear),
        actionButton("Quit", window::close),
        "Ctrl+Enter = Embed · Tab moves to buttons · Esc = Quit"),
      BorderLayout.Location.BOTTOM);

    window.setComponent(root);
    window.addWindowListener(new WindowListenerAdapter() {
      @Override
      public void onResized(
        final Window basePane,
        final TerminalSize oldSize,
        final TerminalSize newSize
      ) {
        result.refresh();
        reflowEditable(input);
      }

      @Override
      public void onUnhandledInput(
        final Window basePane,
        final KeyStroke keyStroke,
        final AtomicBoolean deliverEvent
      ) {
        if (keyStroke.getKeyType() == KeyType.Escape) {
          window.close();
          deliverEvent.set(true);
        } else if (keyStroke.getKeyType() == KeyType.Enter && keyStroke.isCtrlDown()) {
          embed.run();
          deliverEvent.set(true);
        }
      }
    });

    input.takeFocus();
    gui.addWindowAndWait(window);
  }

  private static Component composePanel(
    final TextBox input,
    final Button primary,
    final Button clear,
    final Button quit,
    final String hint
  ) {
    Panel body = new Panel(new LinearLayout(Direction.VERTICAL));
    body.addComponent(new Label("Message"));
    body.addComponent(
      input,
      LinearLayout.createLayoutData(LinearLayout.Alignment.Fill));
    body.addComponent(new EmptySpace(TerminalSize.ONE));
    body.addComponent(
      centeredButtons(primary, clear, quit),
      LinearLayout.createLayoutData(LinearLayout.Alignment.Fill));
    body.addComponent(
      new Label(hint),
      LinearLayout.createLayoutData(LinearLayout.Alignment.Center));
    return body.withBorder(Borders.doubleLine("Compose"));
  }

  private static SoftWrappedPane softPane(final int rows) {
    TextBox box = new TextBox(new TerminalSize(10, rows), TextBox.Style.MULTI_LINE);
    box.setReadOnly(true);
    return new SoftWrappedPane(box);
  }

  private static TextBox editablePane(final int rows) {
    TextBox box = new TextBox(new TerminalSize(10, rows), TextBox.Style.MULTI_LINE);
    AtomicBoolean rewriting = new AtomicBoolean(false);
    box.setTextChangeListener((text, changedByUser) -> {
      if (!changedByUser || !rewriting.compareAndSet(false, true)) {
        return;
      }
      try {
        reflowEditable(box);
      } finally {
        rewriting.set(false);
      }
    });
    return box;
  }

  private static void reflowEditable(final TextBox box) {
    int width = wrapWidth(box);
    String current = box.getText();
    String wrapped = wrapWords(current, width);
    if (wrapped.equals(current)) {
      return;
    }
    var caret = box.getCaretPosition();
    box.setText(wrapped);
    int row = Math.min(caret.getRow(), Math.max(0, box.getLineCount() - 1));
    int column = row < box.getLineCount()
      ? Math.min(caret.getColumn(), box.getLine(row).length())
      : 0;
    box.setCaretPosition(row, column);
  }

  private static Panel centeredButtons(final Button... buttons) {
    Panel row = new Panel(new LinearLayout(Direction.HORIZONTAL));
    for (int i = 0; i < buttons.length; i++) {
      if (i > 0) {
        row.addComponent(new EmptySpace(new TerminalSize(2, 1)));
      }
      // Grow each button so the clickable area is wide; tiny bordered chips are easy to miss.
      row.addComponent(
        buttons[i],
        LinearLayout.createLayoutData(LinearLayout.Alignment.Fill,
          LinearLayout.GrowPolicy.CanGrow));
    }
    return row;
  }

  private static Button actionButton(final String label, final Runnable action) {
    return new ActionButton(label, action);
  }

  private static List<RagChoice> ragChoices(final boolean gtePresent) {
    return List.of(
      new RagChoice(RagMode.NONE, true),
      new RagChoice(RagMode.BM25, true),
      new RagChoice(RagMode.DENSE, gtePresent),
      new RagChoice(RagMode.HYBRID, gtePresent));
  }

  private static int wrapWidth(final TextBox box) {
    int columns = box.getSize().getColumns();
    if (columns <= 1) {
      columns = Math.max(10, box.getPreferredSize().getColumns());
    }
    return Math.max(16, columns - 1);
  }

  private static String wrapWords(final String text, final int width) {
    if (text == null || text.isEmpty()) {
      return "";
    }
    StringBuilder out = new StringBuilder(text.length() + 16);
    String[] paragraphs = text.split("\n", -1);
    for (int p = 0; p < paragraphs.length; p++) {
      if (p > 0) {
        out.append('\n');
      }
      wrapParagraph(paragraphs[p], width, out);
    }
    return out.toString();
  }

  private static void wrapParagraph(
    final String paragraph,
    final int width,
    final StringBuilder out
  ) {
    if (paragraph.isEmpty()) {
      return;
    }
    int cursor = 0;
    while (cursor < paragraph.length()) {
      if (out.length() > 0 && out.charAt(out.length() - 1) != '\n'
        && cursor > 0) {
        out.append('\n');
      }
      int remaining = paragraph.length() - cursor;
      if (remaining <= width) {
        out.append(paragraph, cursor, paragraph.length());
        return;
      }
      int breakAt = paragraph.lastIndexOf(' ', cursor + width);
      if (breakAt < cursor) {
        breakAt = cursor + width;
      }
      out.append(paragraph, cursor, breakAt);
      cursor = breakAt < paragraph.length() && paragraph.charAt(breakAt) == ' '
        ? breakAt + 1
        : breakAt;
    }
  }

  private static String truncate(final String text, final int max) {
    String flat = text.replace('\n', ' ').strip();
    return flat.length() <= max ? flat : flat.substring(0, max - 1) + "…";
  }

  /**
   * Lanterna's stock {@link Button} only fires on {@code CLICK_DOWN} via {@code isActivationStroke}.
   * Combined with a small preferred size, mouse clicks often miss the interactable map cell. This
   * button uses a large preferred size and treats left-button press as an explicit activation.
   */
  private static final class ActionButton extends Button {
    ActionButton(final String label, final Runnable action) {
      super(" " + label + " ", action);
      this.setRenderer(new Button.BorderedButtonRenderer());
    }

    @Override
    public Result handleKeyStroke(final KeyStroke keyStroke) {
      if (keyStroke instanceof com.googlecode.lanterna.input.MouseAction mouse
        && mouse.getActionType() == com.googlecode.lanterna.input.MouseActionType.CLICK_DOWN
        && (mouse.getButton() == 0 || mouse.getButton() == 1)) {
        this.triggerActions();
        return Result.HANDLED;
      }
      return super.handleKeyStroke(keyStroke);
    }
  }

  private static final class SoftWrappedPane {
    private final TextBox box;
    private final StringBuilder raw = new StringBuilder();

    SoftWrappedPane(final TextBox box) {
      this.box = box;
    }

    TextBox component() {
      return this.box;
    }

    void clear() {
      this.raw.setLength(0);
      this.box.setText("");
    }

    void append(final String chunk) {
      if (chunk == null || chunk.isEmpty()) {
        return;
      }
      this.raw.append(chunk);
      this.refresh();
    }

    void appendTurn(final String user) {
      if (this.raw.length() > 0) {
        this.raw.append('\n');
      }
      this.raw.append("you> ").append(user).append("\nassistant> ");
      this.refresh();
    }

    void refresh() {
      this.box.setText(wrapWords(this.raw.toString(), wrapWidth(this.box)));
    }
  }

  private record SetupResult(Path modelPath, RagMode ragMode) {
  }

  private record RagChoice(RagMode mode, boolean enabled) {
    @Override
    public String toString() {
      return this.mode.label();
    }
  }

  private static final class LoadedSession implements AutoCloseable {
    private final LlmModel model;
    private final RagSetup ragSetup;
    private final CausalBundle causal;

    LoadedSession(final LlmModel model, final RagSetup ragSetup, final CausalBundle causal) {
      this.model = model;
      this.ragSetup = ragSetup;
      this.causal = causal;
    }

    LlmModel model() {
      return this.model;
    }

    CausalBundle causal() {
      return this.causal;
    }

    @Override
    public void close() {
      if (this.causal != null) {
        this.causal.close();
      }
      if (this.ragSetup.embeddingModel() != null) {
        this.ragSetup.embeddingModel().close();
      }
      this.model.close();
    }
  }
}
