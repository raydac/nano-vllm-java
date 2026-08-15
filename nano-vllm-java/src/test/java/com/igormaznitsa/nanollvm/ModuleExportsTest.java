package com.igormaznitsa.nanollvm;

import static java.util.Objects.requireNonNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.IOException;
import java.io.InputStream;
import java.lang.module.ModuleDescriptor;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.WildcardType;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

final class ModuleExportsTest {

  private static final String LIBRARY = "com.igormaznitsa.nanollvm";

  private static final Set<String> EXPORTED = Set.of(
    LIBRARY + ".models",
    LIBRARY + ".llm",
    LIBRARY + ".chat",
    LIBRARY + ".rag",
    LIBRARY + ".exceptions",
    LIBRARY + ".tokenizer",
    LIBRARY + ".utils");

  private static final Set<String> HIDDEN = Set.of(
    LIBRARY + ".engine",
    LIBRARY + ".layers",
    LIBRARY + ".tensor",
    LIBRARY + ".internal",
    LIBRARY + ".prompts",
    LIBRARY + ".models.internal",
    LIBRARY + ".models.llmcontainer",
    LIBRARY + ".models.llmarch");

  private static List<Class<?>> publicTypesInExportedPackages()
    throws IOException, URISyntaxException {
    Path classes =
      Path.of(LlmModelFactory.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    if (!Files.isDirectory(classes)) {
      throw new IllegalStateException("expected exploded classes directory, got " + classes);
    }
    try (Stream<Path> walk = Files.walk(classes)) {
      return walk
        .filter(path -> path.getFileName().toString().endsWith(".class"))
        .filter(path -> !path.getFileName().toString().equals("module-info.class"))
        .map(path -> binaryName(classes.relativize(path)))
        .filter(binary -> EXPORTED.contains(packageName(binary)))
        .map(ModuleExportsTest::load)
        .filter(type -> Modifier.isPublic(type.getModifiers()))
        .toList();
    }
  }

  private static Stream<String> leaksFrom(final Class<?> type) {
    List<String> leaks = new ArrayList<>();
    for (Constructor<?> constructor : type.getDeclaredConstructors()) {
      if (isPublished(constructor.getModifiers())) {
        collectLeaks(leaks, member(type, constructor.toGenericString()),
          constructor.getGenericParameterTypes());
        collectLeaks(leaks, member(type, constructor.toGenericString()),
          constructor.getGenericExceptionTypes());
      }
    }
    for (Method method : type.getDeclaredMethods()) {
      if (isPublished(method.getModifiers())) {
        String owner = member(type, method.toGenericString());
        collectLeaks(leaks, owner, method.getGenericReturnType());
        collectLeaks(leaks, owner, method.getGenericParameterTypes());
        collectLeaks(leaks, owner, method.getGenericExceptionTypes());
      }
    }
    for (Field field : type.getDeclaredFields()) {
      if (isPublished(field.getModifiers())) {
        collectLeaks(leaks, member(type, field.toGenericString()), field.getGenericType());
      }
    }
    if (type.isRecord()) {
      for (RecordComponent component : type.getRecordComponents()) {
        collectLeaks(leaks, member(type, component.getName()), component.getGenericType());
      }
    }
    return leaks.stream();
  }

  private static void collectLeaks(final List<String> leaks, final String owner,
                                   final Type... types) {
    for (Type type : types) {
      hiddenLibraryTypes(type).forEach(
        hidden -> leaks.add(owner + " uses hidden " + hidden.getTypeName()));
    }
  }

  private static Stream<Class<?>> hiddenLibraryTypes(final Type type) {
    return switch (type) {
      case null -> Stream.empty();
      case Class<?> raw -> hiddenFromClass(raw);
      case ParameterizedType parameterized -> Stream.concat(
        hiddenLibraryTypes(parameterized.getRawType()),
        Stream.of(parameterized.getActualTypeArguments())
          .flatMap(ModuleExportsTest::hiddenLibraryTypes));
      case GenericArrayType array -> hiddenLibraryTypes(array.getGenericComponentType());
      case WildcardType wildcard -> Stream.concat(
        Stream.of(wildcard.getUpperBounds()).flatMap(ModuleExportsTest::hiddenLibraryTypes),
        Stream.of(wildcard.getLowerBounds()).flatMap(ModuleExportsTest::hiddenLibraryTypes));
      case TypeVariable<?> variable -> Stream.of(variable.getBounds())
        .flatMap(ModuleExportsTest::hiddenLibraryTypes);
      default -> Stream.empty();
    };
  }

  private static Stream<Class<?>> hiddenFromClass(final Class<?> type) {
    if (type.isArray()) {
      return hiddenFromClass(type.getComponentType());
    }
    if (!type.getName().startsWith(LIBRARY + ".")) {
      return Stream.empty();
    }
    return EXPORTED.contains(type.getPackageName()) ? Stream.empty() : Stream.of(type);
  }

  private static boolean isPublished(final int modifiers) {
    return Modifier.isPublic(modifiers) || Modifier.isProtected(modifiers);
  }

  private static String member(final Class<?> type, final String signature) {
    return type.getTypeName() + ": " + signature;
  }

  private static String binaryName(final Path relative) {
    return relative.toString().replace('\\', '/').replace('/', '.').replaceAll("\\.class$", "");
  }

  private static String packageName(final String binaryName) {
    int nested = binaryName.indexOf('$');
    String top = nested < 0 ? binaryName : binaryName.substring(0, nested);
    int dot = top.lastIndexOf('.');
    return dot < 0 ? "" : top.substring(0, dot);
  }

  private static Class<?> load(final String binaryName) {
    try {
      return Class.forName(binaryName, false, LlmModelFactory.class.getClassLoader());
    } catch (ClassNotFoundException ex) {
      throw new IllegalStateException(binaryName, ex);
    }
  }

  @Test
  void moduleDescriptorExportsOnlyApplicationPackages() throws IOException {
    try (InputStream bytes = requireNonNull(
      LlmModelFactory.class.getResourceAsStream("/module-info.class"),
      "module-info.class")) {
      ModuleDescriptor descriptor = ModuleDescriptor.read(bytes);
      assertEquals(LIBRARY, descriptor.name());
      assertEquals(
        EXPORTED,
        descriptor.exports().stream().map(ModuleDescriptor.Exports::source)
          .collect(Collectors.toSet()));
    }
  }

  @Test
  void applicationEntryPointsArePublicAndExported() {
    Stream.of(
        LlmModelFactory.class,
        ModelSupport.class,
        LLM.class,
        ChatSession.class,
        RagFactory.class,
        Tokenizer.class,
        GgufTokenizerSource.class,
        ResourceLimits.class,
        UnsupportedModelException.class)
      .forEach(type -> {
        assertTrue(Modifier.isPublic(type.getModifiers()), type.getName());
        assertTrue(EXPORTED.contains(type.getPackageName()), type.getName());
      });
  }

  @Test
  void loadAndEnginePackagesStayHiddenFromTheModuleSurface() {
    HIDDEN.forEach(pkg -> assertFalse(EXPORTED.contains(pkg), pkg));
  }

  @Test
  void exportedPublicSignaturesDoNotMentionHiddenPackages() throws IOException, URISyntaxException {
    List<String> leaks = publicTypesInExportedPackages().stream()
      .flatMap(ModuleExportsTest::leaksFrom)
      .toList();
    assertTrue(leaks.isEmpty(), () -> String.join(System.lineSeparator(), leaks));
  }
}
