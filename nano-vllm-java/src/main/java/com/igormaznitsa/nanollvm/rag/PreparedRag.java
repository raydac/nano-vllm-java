package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Immutable, shareable RAG corpus: preparsed passages + inverted BM25 index.
 * Load once via {@link RagFactory}; reuse across many LLM sessions.
 *
 * <p>Safe to share across threads after construction. {@link #retrieve} returns an unmodifiable
 * list.
 */
public final class PreparedRag implements RagIndex {

  private static final double SHORT_PASSAGE_CHARS = 200.0;
  private static final double K1 = 1.2;
  private static final double B = 0.75;

  private final List<Passage> passages;
  private final List<TextChunk> chunks;
  private final Map<String, List<Posting>> inverted;
  private final Map<String, Double> idf;
  private final Map<String, Integer> docFreq;
  private final double avgDocLen;
  private final int docCount;
  private final Path sourceRoot;
  private final RagLoadOptions options;

  private PreparedRag(
    final List<Passage> passages,
    final Map<String, List<Posting>> inverted,
    final Map<String, Double> idf,
    final Map<String, Integer> docFreq,
    final double avgDocLen,
    final Path sourceRoot,
    final RagLoadOptions options
  ) {
    this.passages = List.copyOf(requireNonNull(passages, "passages"));
    this.inverted = Map.copyOf(inverted);
    this.idf = Map.copyOf(idf);
    this.docFreq = Map.copyOf(docFreq);
    this.avgDocLen = avgDocLen;
    this.docCount = this.passages.size();
    this.sourceRoot = sourceRoot;
    this.options = requireNonNull(options, "options");
    this.chunks = this.passages.stream().map(Passage::chunk).toList();
  }

  static PreparedRag fromChunks(
    final List<TextChunk> chunks,
    final Path sourceRoot,
    final RagLoadOptions options
  ) {
    requireNonNull(chunks, "chunks");
    requireNonNull(options, "options");
    List<Passage> prepared = PassagePrep.prepare(chunks);
    return buildIndex(prepared, sourceRoot, options);
  }

  static List<String> tokenize(final String text) {
    return Lexicon.tokenize(text);
  }

  /**
   * Fraction of distinct query terms that appear in the passage (0..1).
   * Favors passages that mention more of the query over long text that only
   * shares a title token.
   */
  static double termCoverage(final String passageText, final List<String> queryTerms) {
    if (queryTerms.isEmpty()) {
      return 0.0;
    }
    Set<String> passageTerms = new LinkedHashSet<>(PreparedRag.tokenize(passageText));
    long hit = queryTerms.stream().filter(passageTerms::contains).count();
    return hit / (double) queryTerms.size();
  }

  /**
   * Mild length density: shorter passages score higher when coverage is equal.
   * Corpus-agnostic — no filename or topic rules.
   */
  static double groundedScore(final RagHit hit, final List<String> queryTerms) {
    double coverage = 1.0 + termCoverage(hit.chunk().text(), queryTerms);
    int len = Math.max(hit.chunk().text().length(), 1);
    double density = 1.0 + (SHORT_PASSAGE_CHARS / (SHORT_PASSAGE_CHARS + len));
    return hit.score() * coverage * density;
  }

  private static PreparedRag buildIndex(
    final List<Passage> passages,
    final Path sourceRoot,
    final RagLoadOptions options
  ) {
    if (passages.isEmpty()) {
      throw new IllegalArgumentException("passages must not be empty");
    }

    Map<String, List<Posting>> inverted = new HashMap<>();
    Map<String, Integer> docFreq = new HashMap<>();
    long totalTokens = 0L;

    for (int docId = 0; docId < passages.size(); docId++) {
      Passage passage = passages.get(docId);
      totalTokens += passage.tokenCount();
      for (Map.Entry<String, Integer> entry : passage.termFreqs().entrySet()) {
        String term = entry.getKey();
        docFreq.merge(term, 1, Integer::sum);
        inverted.computeIfAbsent(term, key -> new ArrayList<>())
          .add(new Posting(docId, entry.getValue()));
      }
    }

    Map<String, Double> idf = HashMap.newHashMap(docFreq.size());
    int n = passages.size();
    for (Map.Entry<String, Integer> entry : docFreq.entrySet()) {
      int df = entry.getValue();
      idf.put(entry.getKey(), Math.log(1.0 + (n - df + 0.5) / (df + 0.5)));
    }

    Map<String, List<Posting>> frozenPostings = HashMap.newHashMap(inverted.size());
    for (Map.Entry<String, List<Posting>> entry : inverted.entrySet()) {
      frozenPostings.put(entry.getKey(), List.copyOf(entry.getValue()));
    }

    double avg = totalTokens / (double) passages.size();
    return new PreparedRag(
      passages,
      frozenPostings,
      idf,
      Map.copyOf(docFreq),
      Math.max(avg, 1.0),
      sourceRoot,
      options
    );
  }

  private static List<RagHit> keepStrongHits(final List<RagHit> scored, final int topK) {
    if (scored.isEmpty()) {
      return List.of();
    }
    double best = scored.getFirst().score();
    double floor = best * 0.45;
    return scored.stream()
      .filter(hit -> hit.score() >= floor)
      .limit(topK)
      .toList();
  }

  List<Passage> passages() {
    return this.passages;
  }

  public List<TextChunk> chunks() {
    return this.chunks;
  }

  public Optional<Path> sourceRoot() {
    return Optional.ofNullable(this.sourceRoot);
  }

  public RagLoadOptions options() {
    return this.options;
  }

  @Override
  public int size() {
    return this.docCount;
  }

  @Override
  public boolean isOutsideCorpus(final String query) {
    List<String> raw = List.copyOf(new LinkedHashSet<>(Lexicon.tokenizeSurface(query)));
    return QueryTerms.queryOutsideCorpus(this.docFreq, this.docCount, raw);
  }

  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    List<String> terms = this.selectedQueryTerms(query);
    if (terms.isEmpty()) {
      return List.of();
    }
    List<RagHit> hits = this.bm25Retrieve(query, Math.max(topK * 4, topK));
    return hits.stream()
      .map(hit -> new RagHit(hit.chunk(), groundedScore(hit, terms)))
      .filter(hit -> termCoverage(hit.chunk().text(), terms) > 0.0)
      .sorted(Comparator
        .comparingDouble(RagHit::score).reversed()
        .thenComparingInt(hit -> hit.chunk().text().length()))
      .limit(topK)
      .toList();
  }

  @Override
  public String toString() {
    return "PreparedRag{passages=%d, source=%s}".formatted(
      this.size(),
      this.sourceRoot == null ? "inline" : this.sourceRoot);
  }

  private List<String> selectedQueryTerms(final String query) {
    return QueryTerms.select(this.docFreq, this.docCount, query);
  }

  private List<RagHit> bm25Retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    List<String> terms = QueryTerms.select(this.docFreq, this.docCount, query);
    if (terms.isEmpty()) {
      return List.of();
    }
    int rawDistinct = new LinkedHashSet<>(Lexicon.tokenizeSurface(query)).size();
    if (QueryTerms.queryTooBroadForCorpus(rawDistinct, terms)) {
      return List.of();
    }

    Set<Integer> candidates = new LinkedHashSet<>();
    for (String term : terms) {
      List<Posting> postings = this.inverted.get(term);
      if (postings != null) {
        for (Posting posting : postings) {
          candidates.add(posting.docId());
        }
      }
    }
    if (candidates.isEmpty()) {
      return List.of();
    }

    List<RagHit> scored = new ArrayList<>(candidates.size());
    for (int docId : candidates) {
      Passage passage = this.passages.get(docId);
      if (!QueryTerms.qualifies(passage, terms)) {
        continue;
      }
      double score = this.scoreDocument(docId, terms);
      if (score > 0.0) {
        scored.add(new RagHit(passage.chunk(), score));
      }
    }
    scored.sort(Comparator.comparingDouble(RagHit::score).reversed());
    return List.copyOf(keepStrongHits(scored, topK));
  }

  private double scoreDocument(final int docIndex, final List<String> queryTerms) {
    Passage passage = this.passages.get(docIndex);
    Map<String, Integer> tf = passage.termFreqs();
    int docLen = passage.tokenCount();
    double score = 0.0;
    for (String term : queryTerms) {
      int freq = tf.getOrDefault(term, 0);
      if (freq == 0) {
        continue;
      }
      double termIdf = this.idf.getOrDefault(term, 0.0);
      double denom = freq + K1 * (1.0 - B + B * docLen / this.avgDocLen);
      score += termIdf * (freq * (K1 + 1.0)) / denom;
    }
    return score;
  }

  private record Posting(int docId, int tf) {
  }

  record Passage(
    TextChunk chunk,
    String searchText,
    Map<String, Integer> termFreqs,
    int tokenCount
  ) {

    public Passage {
      requireNonNull(chunk, "chunk");
      requireNonNull(searchText, "searchText");
      termFreqs = Map.copyOf(requireNonNull(termFreqs, "termFreqs"));
      if (tokenCount < 0) {
        throw new IllegalArgumentException("tokenCount must be >= 0");
      }
    }

    public String id() {
      return this.chunk.id();
    }

    public String source() {
      return this.chunk.source();
    }

    public String modelText() {
      return this.chunk.text();
    }
  }

  private static final class Lexicon {

    private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");
    private static final Pattern STEM_SPLIT = Pattern.compile("[^\\p{L}\\p{N}]+");

    private Lexicon() {
    }

    static List<String> tokenize(final String text) {
      List<String> tokens = new ArrayList<>();
      var matcher = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
      while (matcher.find()) {
        String token = matcher.group();
        if (token.length() <= 1) {
          continue;
        }
        tokens.add(token);
        addInflectionKeys(token, tokens);
      }
      return tokens;
    }

    static List<String> tokenizeSurface(final String text) {
      List<String> tokens = new ArrayList<>();
      var matcher = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
      while (matcher.find()) {
        String token = matcher.group();
        if (token.length() > 1) {
          tokens.add(token);
        }
      }
      return tokens;
    }

    static Map<String, Integer> termFrequencies(final String text) {
      Map<String, Integer> tf = new HashMap<>();
      for (String token : tokenize(text)) {
        tf.merge(token, 1, Integer::sum);
      }
      return Map.copyOf(tf);
    }

    private static void addInflectionKeys(final String token, final List<String> tokens) {
      if (!isCyrillicToken(token)) {
        return;
      }
      if (token.length() >= 3) {
        tokens.add(token.substring(0, token.length() - 1));
      }
      if (token.length() >= 5) {
        tokens.add(token.substring(0, token.length() - 2));
      }
      if (token.length() > 5) {
        tokens.add(token.substring(0, 5));
      }
    }

    private static boolean isCyrillicToken(final String token) {
      return token.chars().anyMatch(c -> c >= 0x0400 && c <= 0x04FF);
    }

    static String normalizeModelText(final String raw) {
      if (raw == null || raw.isBlank()) {
        return "";
      }
      String text = Normalizer.normalize(raw, Normalizer.Form.NFC);
      return text.replace('\u00A0', ' ').replaceAll("\\s+", " ").strip();
    }

    static String buildSearchText(
      final String modelText,
      final String source,
      final boolean includeSourceStems
    ) {
      StringBuilder search = new StringBuilder(stripSectionPrefix(modelText));
      if (includeSourceStems) {
        for (String stemToken : sourceStemTokens(source)) {
          search.append(' ').append(stemToken);
        }
      }
      return search.toString();
    }

    static String stripSectionPrefix(final String modelText) {
      if (modelText == null || modelText.isBlank()) {
        return "";
      }
      int sep = modelText.indexOf(" — ");
      if (sep > 0 && sep <= 80) {
        return modelText.substring(sep + 3).strip();
      }
      return modelText;
    }

    static List<String> sourceStemTokens(final String source) {
      if (source == null || source.isBlank()) {
        return List.of();
      }
      String name = Lexicon.fileNameOnly(source);
      int dot = name.lastIndexOf('.');
      if (dot > 0) {
        name = name.substring(0, dot);
      }
      name = name.toLowerCase(Locale.ROOT);
      List<String> tokens = new ArrayList<>();
      for (String part : STEM_SPLIT.split(name, -1)) {
        if (part.length() > 1) {
          tokens.add(part);
        }
      }
      if (tokens.isEmpty() && name.length() > 1) {
        tokens.add(name);
      }
      return List.copyOf(tokens);
    }

    static String fileNameOnly(final String source) {
      String path =
        source.startsWith("classpath:") ? source.substring("classpath:".length()) : source;
      int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
      return slash >= 0 ? path.substring(slash + 1) : path;
    }
  }

  private static final class PassagePrep {

    private PassagePrep() {
    }

    static List<Passage> prepare(final List<TextChunk> chunks) {
      requireNonNull(chunks, "chunks");
      List<Passage> prepared = new ArrayList<>(chunks.size());
      Set<String> sourcesWithStems = new LinkedHashSet<>();
      for (TextChunk chunk : chunks) {
        if (!chunk.isBlank()) {
          prepared.add(prepareOne(chunk, sourcesWithStems.add(chunk.source())));
        }
      }
      if (prepared.isEmpty()) {
        throw new IllegalArgumentException("no non-blank passages to prepare");
      }
      return List.copyOf(prepared);
    }

    private static Passage prepareOne(final TextChunk chunk, final boolean includeSourceStems) {
      requireNonNull(chunk, "chunk");
      String modelText = Lexicon.normalizeModelText(chunk.text());
      TextChunk normalized = new TextChunk(chunk.id(), chunk.source(), modelText);
      String searchText = Lexicon.buildSearchText(modelText, chunk.source(), includeSourceStems);
      Map<String, Integer> tf = Lexicon.termFrequencies(searchText);
      int tokens = tf.values().stream().mapToInt(Integer::intValue).sum();
      return new Passage(normalized, searchText, tf, tokens);
    }
  }

  private static final class QueryTerms {

    private static final int MAX_SELECTED_TERMS = 5;
    private static final int CONTENTFUL_MIN_LEN = 3;

    private static final Set<String> QUERY_GLUE = Set.of(
      "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as", "at", "by", "for",
      "from", "in", "into", "of", "on", "to", "with", "without", "about", "over", "under", "up",
      "down", "out", "off", "not", "no", "yes", "do", "does", "did", "doing", "done", "be", "am",
      "is", "are", "was", "were", "been", "being", "have", "has", "had", "having", "can", "could",
      "may", "might", "must", "shall", "should", "will", "would", "i", "me", "my", "we", "our",
      "you", "your", "he", "she", "it", "they", "them", "their", "this", "that", "these", "those",
      "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "please", "tell",
      "say", "said", "ask", "think", "know", "like", "want", "need", "get", "got", "just", "also",
      "only", "very", "too", "more", "most", "some", "any", "all", "own", "same", "other", "such"
    );

    private QueryTerms() {
    }

    static List<String> select(
      final Map<String, Integer> docFreq,
      final int docCount,
      final String query
    ) {
      requireNonNull(docFreq, "docFreq");
      requireNonNull(query, "query");
      if (docCount <= 0) {
        return List.of();
      }
      List<String> surface = List.copyOf(new LinkedHashSet<>(Lexicon.tokenizeSurface(query)));
      if (surface.isEmpty()) {
        return List.of();
      }
      if (queryOutsideCorpus(docFreq, docCount, surface)) {
        return List.of();
      }

      List<String> expanded = List.copyOf(new LinkedHashSet<>(PreparedRag.tokenize(query)));
      List<String> known = expanded.stream()
        .filter(QueryTerms::isContentful)
        .filter(term -> docFreq.getOrDefault(term, 0) > 0)
        .toList();
      if (known.isEmpty()) {
        return List.of();
      }

      return distinctiveKnown(known).stream()
        .sorted(Comparator
          .comparingInt(String::length).reversed()
          .thenComparingDouble((String term) -> -idfWeight(docFreq, docCount, term)))
        .limit(MAX_SELECTED_TERMS)
        .toList();
    }

    static boolean queryOutsideCorpus(
      final Map<String, Integer> docFreq,
      final int docCount,
      final List<String> surfaceTerms
    ) {
      requireNonNull(docFreq, "docFreq");
      requireNonNull(surfaceTerms, "surfaceTerms");
      if (surfaceTerms.isEmpty() || docCount <= 0) {
        return false;
      }
      List<String> contentful = surfaceTerms.stream()
        .filter(QueryTerms::isSalientContentful)
        .toList();
      if (contentful.isEmpty()) {
        return false;
      }
      long oov = contentful.stream().filter(term -> isContentfulOov(docFreq, term)).count();
      if (oov == 0L) {
        return false;
      }
      if (oov * 2L >= contentful.size()) {
        return true;
      }
      return hasLongestContentfulOov(docFreq, contentful);
    }

    static boolean queryTooBroadForCorpus(
      final int rawDistinctTerms,
      final List<String> selectedTerms
    ) {
      return rawDistinctTerms >= 3 && selectedTerms.isEmpty();
    }

    static boolean qualifies(final Passage passage, final List<String> selectedTerms) {
      requireNonNull(passage, "passage");
      requireNonNull(selectedTerms, "selectedTerms");
      if (selectedTerms.isEmpty()) {
        return false;
      }
      Set<String> passageTerms = new LinkedHashSet<>(passage.termFreqs().keySet());
      long matched = selectedTerms.stream().filter(passageTerms::contains).count();
      int need = Math.clamp((selectedTerms.size() + 1) / 2, 1, 2);
      return matched >= need;
    }

    private static double idfWeight(
      final Map<String, Integer> docFreq,
      final int docCount,
      final String term
    ) {
      int df = docFreq.getOrDefault(term, 0);
      if (df <= 0) {
        return 0.0;
      }
      return Math.log(1.0 + (docCount - df + 0.5) / (df + 0.5));
    }

    private static boolean isContentful(final String term) {
      return term != null && term.length() >= CONTENTFUL_MIN_LEN;
    }

    private static boolean isSalientContentful(final String term) {
      return isContentful(term) && !QUERY_GLUE.contains(term);
    }

    private static List<String> distinctiveKnown(final List<String> known) {
      boolean hasLong = known.stream().anyMatch(term -> term.length() >= 5);
      if (!hasLong) {
        return known;
      }
      List<String> longer = known.stream().filter(term -> term.length() >= 5).toList();
      return longer.isEmpty() ? known : longer;
    }

    private static boolean hasLongestContentfulOov(
      final Map<String, Integer> docFreq,
      final List<String> contentful
    ) {
      int maxLen = contentful.stream().mapToInt(String::length).max().orElse(0);
      return contentful.stream()
        .filter(term -> term.length() == maxLen)
        .anyMatch(term -> isContentfulOov(docFreq, term));
    }

    private static boolean isContentfulOov(final Map<String, Integer> docFreq, final String term) {
      if (!isContentful(term) || docFreq.getOrDefault(term, 0) > 0) {
        return false;
      }
      return PreparedRag.tokenize(term).stream()
        .noneMatch(key -> docFreq.getOrDefault(key, 0) > 0);
    }
  }
}
