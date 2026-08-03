package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.Config;
import com.igormaznitsa.nanollvm.EngineIo;
import com.igormaznitsa.nanollvm.ModelLoadException;
import com.igormaznitsa.nanollvm.layers.Sampler;
import com.igormaznitsa.nanollvm.models.CausalLM;
import com.igormaznitsa.nanollvm.models.CausalLMFactory;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import com.igormaznitsa.nanollvm.utils.Context;
import com.igormaznitsa.nanollvm.utils.ModelLoader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class ModelRunner implements AutoCloseable {

  private final Config config;
  private final EngineIo io;
  private final int blockSize;
  private final CausalLM model;
  private final Sampler sampler = new Sampler();

  public ModelRunner(Config config) {
    this(config, EngineIo.silent());
  }

  public ModelRunner(Config config, EngineIo io) {
    this.config = config;
    this.io = io == null ? EngineIo.silent() : io;
    this.blockSize = config.kvcacheBlockSize();

    long t0 = System.nanoTime();
    this.io.info("CPU backend: " + VectorMath.backendInfo());
    String arch = CausalLMFactory.detect(config.hfConfig());
    this.io.info("Building " + arch + " model graph…");
    this.model = CausalLMFactory.create(config.hfConfig());
    this.io.infof("Model graph ready (%s) in %.1fs%n",
        this.model.architectureName(), (System.nanoTime() - t0) / 1e9);

    try {
      ModelLoader.loadModel(this.model, config.model(), this.io);
    } catch (IOException e) {
      throw new ModelLoadException("failed to load model weights from " + config.model(), e);
    }

    this.io.info("Allocating KV cache…");
    long tKv = System.nanoTime();
    this.allocateKvCache();
    this.io.infof("KV cache ready: %d blocks (%.1fs)%n",
        this.config.numKvcacheBlocks(),
        (System.nanoTime() - tKv) / 1e9);
    this.io.infof("Model runner ready in %.1fs%n", (System.nanoTime() - t0) / 1e9);
  }

  private static Tensor toTensor1d(List<Integer> values) {
    float[] data = new float[values.size()];
    for (int i = 0; i < values.size(); i++) {
      data[i] = values.get(i);
    }
    return Tensor.of(data, values.size());
  }

  public List<Integer> run(List<Sequence> seqs, boolean isPrefill) {
    PreparedInputs prepared = this.prepareInputs(seqs, isPrefill);
    SamplingControls sampling = this.collectSamplingControls(seqs);

    Tensor hidden = this.model.forward(prepared.inputIds(), prepared.positions());
    Tensor logits = this.model.computeLogits(hidden);
    int[] sampledIds = this.sampler.forward(
        logits, sampling.temperatures(), sampling.topKs(), sampling.topPs());

    Context.reset();
    return this.toTokenIdList(sampledIds);
  }

  private PreparedInputs prepareInputs(List<Sequence> seqs, boolean isPrefill) {
    return isPrefill ? this.preparePrefill(seqs) : this.prepareDecode(seqs);
  }

  private SamplingControls collectSamplingControls(List<Sequence> seqs) {
    float[] temperatures = new float[seqs.size()];
    int[] topKs = new int[seqs.size()];
    float[] topPs = new float[seqs.size()];
    for (int i = 0; i < seqs.size(); i++) {
      Sequence seq = seqs.get(i);
      temperatures[i] = seq.temperature();
      topKs[i] = seq.topK();
      topPs[i] = seq.topP();
    }
    return new SamplingControls(temperatures, topKs, topPs);
  }

  private List<Integer> toTokenIdList(int[] tokenIds) {
    return Arrays.stream(tokenIds).boxed().toList();
  }

  @Override
  public void close() {
    Context.reset();
  }

  private void allocateKvCache() {
    Config.HfConfig hf = this.config.hfConfig();
    if (this.config.numKvcacheBlocks() <= 0) {
      int blocksPerSeq = (this.config.maxModelLen() + this.blockSize - 1) / this.blockSize;
      int estimated = Math.max(this.config.maxNumSeqs() * blocksPerSeq, 128);
      long free = Runtime.getRuntime().maxMemory();
      long bytesPerBlock = 2L * hf.numHiddenLayers() * this.blockSize
          * hf.numKeyValueHeads() * hf.headDim() * Float.BYTES;
      int heapCap = (int) Math.max(32, (free / 4) / Math.max(1, bytesPerBlock));
      this.config.setNumKvcacheBlocks(Math.min(estimated, heapCap));
    }
    if (this.config.numKvcacheBlocks() <= 0) {
      throw new IllegalStateException("numKvcacheBlocks must be > 0");
    }
    this.model.attentionLayers().forEach(attn -> {
      Tensor k = Tensor.zeros(this.config.numKvcacheBlocks(), this.blockSize, hf.numKeyValueHeads(),
          hf.headDim());
      Tensor v = Tensor.zeros(this.config.numKvcacheBlocks(), this.blockSize, hf.numKeyValueHeads(),
          hf.headDim());
      attn.setCaches(k, v);
    });
  }

  private int[][] prepareBlockTables(List<Sequence> seqs) {
    int maxLen = seqs.stream().mapToInt(seq -> seq.blockTable().size()).max().orElse(0);
    int[][] tables = new int[seqs.size()][maxLen];
    for (int i = 0; i < seqs.size(); i++) {
      List<Integer> table = seqs.get(i).blockTable();
      for (int j = 0; j < maxLen; j++) {
        tables[i][j] = j < table.size() ? table.get(j) : -1;
      }
    }
    return tables;
  }

  private PreparedInputs preparePrefill(List<Sequence> seqs) {
    List<Integer> inputIds = new ArrayList<>();
    List<Integer> positions = new ArrayList<>();
    List<Integer> cuSeqlensQ = new ArrayList<>();
    List<Integer> cuSeqlensK = new ArrayList<>();
    cuSeqlensQ.add(0);
    cuSeqlensK.add(0);
    int maxSeqlenQ = 0;
    int maxSeqlenK = 0;
    List<Integer> slotMapping = new ArrayList<>();
    boolean anyBlockTable = false;

    for (Sequence seq : seqs) {
      int start = seq.numCachedTokens();
      int seqlenQ = seq.numScheduledTokens();
      int end = start + seqlenQ;
      int seqlenK = end;
      for (int i = start; i < end; i++) {
        inputIds.add(seq.tokenAt(i));
        positions.add(i);
      }
      cuSeqlensQ.add(cuSeqlensQ.get(cuSeqlensQ.size() - 1) + seqlenQ);
      cuSeqlensK.add(cuSeqlensK.get(cuSeqlensK.size() - 1) + seqlenK);
      maxSeqlenQ = Math.max(maxSeqlenQ, seqlenQ);
      maxSeqlenK = Math.max(maxSeqlenK, seqlenK);
      if (seq.blockTable().isEmpty()) {
        continue;
      }
      anyBlockTable = true;
      int startBlock = start / this.blockSize;
      int endBlock = (end + this.blockSize - 1) / this.blockSize;
      for (int i = startBlock; i < endBlock; i++) {
        int slotStart = seq.blockTable().get(i) * this.blockSize;
        if (i == startBlock) {
          slotStart += start % this.blockSize;
        }
        int slotEnd;
        if (i != endBlock - 1) {
          slotEnd = seq.blockTable().get(i) * this.blockSize + this.blockSize;
        } else {
          slotEnd = seq.blockTable().get(i) * this.blockSize + end - i * this.blockSize;
        }
        for (int s = slotStart; s < slotEnd; s++) {
          slotMapping.add(s);
        }
      }
    }

    int[][] blockTables = null;
    if (cuSeqlensK.get(cuSeqlensK.size() - 1) > cuSeqlensQ.get(cuSeqlensQ.size() - 1) &&
        anyBlockTable) {
      blockTables = this.prepareBlockTables(seqs);
    }

    Context.set(
        true,
        cuSeqlensQ.stream().mapToInt(Integer::intValue).toArray(),
        cuSeqlensK.stream().mapToInt(Integer::intValue).toArray(),
        maxSeqlenQ,
        maxSeqlenK,
        slotMapping.stream().mapToInt(Integer::intValue).toArray(),
        null,
        blockTables
    );
    return new PreparedInputs(toTensor1d(inputIds), toTensor1d(positions));
  }

  private PreparedInputs prepareDecode(List<Sequence> seqs) {
    int n = seqs.size();
    float[] inputIds = new float[n];
    float[] positions = new float[n];
    int[] slotMapping = new int[n];
    int[] contextLens = new int[n];
    for (int i = 0; i < n; i++) {
      Sequence seq = seqs.get(i);
      inputIds[i] = seq.lastToken();
      positions[i] = seq.length() - 1;
      contextLens[i] = seq.length();
      slotMapping[i] = seq.blockTable().get(seq.blockTable().size() - 1) * this.blockSize
          + seq.lastBlockNumTokens() - 1;
    }
    Context.set(false, null, null, 0, 0, slotMapping, contextLens, this.prepareBlockTables(seqs));
    return new PreparedInputs(Tensor.of(inputIds, n), Tensor.of(positions, n));
  }

  private record PreparedInputs(Tensor inputIds, Tensor positions) {
  }

  private record SamplingControls(float[] temperatures, int[] topKs, float[] topPs) {
  }
}
