package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Okapi BM25 over {@link PreparedPassage}s with an inverted index built at load time.
 * Queries score only candidate docs that contain at least one query term.
 */
public final class Bm25Index implements RagIndex {

  private static final double K1 = 1.2;
  private static final double B = 0.75;

  private final List<PreparedPassage> passages;
  private final Map<String, List<Posting>> inverted;
  private final Map<String, Double> idf;
  private final Map<String, Integer> docFreq;
  private final double avgDocLen;
  private final int docCount;

  private Bm25Index(
      List<PreparedPassage> passages,
      Map<String, List<Posting>> inverted,
      Map<String, Double> idf,
      Map<String, Integer> docFreq,
      double avgDocLen
  ) {
    this.passages = List.copyOf(passages);
    this.inverted = Map.copyOf(inverted);
    this.idf = Map.copyOf(idf);
    this.docFreq = Map.copyOf(docFreq);
    this.avgDocLen = avgDocLen;
    this.docCount = passages.size();
  }

  public static Bm25Index build(TextCorpus corpus) {
    requireNonNull(corpus, "corpus");
    if (corpus.isEmpty()) {
      throw new IllegalArgumentException("corpus must not be empty");
    }
    return buildPrepared(PassagePreparser.prepare(corpus.chunks()));
  }

  public static Bm25Index buildPrepared(List<PreparedPassage> passages) {
    requireNonNull(passages, "passages");
    if (passages.isEmpty()) {
      throw new IllegalArgumentException("passages must not be empty");
    }

    Map<String, List<Posting>> inverted = new HashMap<>();
    Map<String, Integer> docFreq = new HashMap<>();
    long totalTokens = 0L;

    for (int docId = 0; docId < passages.size(); docId++) {
      PreparedPassage passage = passages.get(docId);
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
    return new Bm25Index(passages, frozenPostings, idf, Map.copyOf(docFreq), Math.max(avg, 1.0));
  }

  public static Bm25Index of(String... texts) {
    return build(TextCorpus.ofStrings(texts));
  }

  public static Bm25Index fromFile(Path file) {
    return build(TextCorpus.fromFile(file));
  }

  public static Bm25Index fromFolder(Path folder) {
    return build(TextCorpus.fromFolder(folder));
  }

  /**
   * Drops hits far below the best score so weak lexical matches do not dilute the prompt.
   */
  static List<RagHit> keepStrongHits(List<RagHit> scored, int topK) {
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

  static List<String> tokenize(String text) {
    return PassagePreparser.tokenize(text);
  }

  List<String> selectedQueryTerms(String query) {
    return RagQueryTerms.select(this.docFreq, this.docCount, query);
  }

  boolean isOutsideCorpus(String query) {
    List<String> raw = List.copyOf(new LinkedHashSet<>(PassagePreparser.tokenize(query)));
    return RagQueryTerms.queryOutsideCorpus(this.docFreq, raw);
  }

  public List<PreparedPassage> passages() {
    return this.passages;
  }

  public TextCorpus corpus() {
    return TextCorpus.ofChunks(this.passages.stream().map(PreparedPassage::chunk).toList());
  }

  @Override
  public int size() {
    return this.docCount;
  }

  @Override
  public List<RagHit> retrieve(String query, int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    List<String> terms = RagQueryTerms.select(this.docFreq, this.docCount, query);
    if (terms.isEmpty()) {
      return List.of();
    }
    int rawDistinct = new LinkedHashSet<>(PassagePreparser.tokenize(query)).size();
    if (RagQueryTerms.queryTooBroadForCorpus(rawDistinct, terms)) {
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
      PreparedPassage passage = this.passages.get(docId);
      if (!RagQueryTerms.qualifies(passage, terms)) {
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

  private double scoreDocument(int docIndex, List<String> queryTerms) {
    PreparedPassage passage = this.passages.get(docIndex);
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
}
