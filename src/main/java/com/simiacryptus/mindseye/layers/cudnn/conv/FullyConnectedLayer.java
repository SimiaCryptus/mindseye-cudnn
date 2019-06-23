/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.layers.java.FullyConnectedReferenceLayer;
import com.simiacryptus.mindseye.layers.java.ReshapeLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;

@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase implements MultiPrecision<FullyConnectedLayer>, Explodable {
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  @Nullable
  public final int[] inputDims;
  @Nullable
  public final int[] outputDims;
  @Nullable
  private final Tensor weights;

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private int batchBands = 0;

  private FullyConnectedLayer() {
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outs);
    setWeights(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      final double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }

  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    @Nullable final Tensor data = Tensor.fromJson(json.get("weights"), rs);
    weights = data;
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  @Override
  protected void _free() {
    weights.freeRef();
    super._free();
  }

  @Nonnull
  public FullyConnectedLayer set(final double[] data) {
    weights.set(data);
    return this;
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final Tensor data) {
    weights.set(data);
    return this;
  }

  @Nonnull
  public FullyConnectedLayer setWeightsLog(final double value) {
    getWeights().setByCoord(c -> (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    return this;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return new FullyConnectedReferenceLayer(inputDims, outputDims).set(getWeights());
  }

  @Nullable
  @Override
  public Result evalAndFree(final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    Layer explode = explode();
    Result eval = explode.evalAndFree(inObj);
    explode.freeRef();
    return eval;
  }

  public Layer explodeAndFree() {
    Layer explode = explode();
    freeRef();
    return explode;
  }

  @Nonnull
  public Layer explode() {
    int inputVol = Tensor.length(inputDims);
    int outVol = Tensor.length(outputDims);
    @Nonnull PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(new ReshapeLayer(1, 1, inputVol)).freeRef();
    @Nullable Tensor tensor = this.weights.reshapeCast(1, 1, inputVol * outVol);
    @Nonnull ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, inputVol, outVol)
        .set(tensor)
        .setBatchBands(getBatchBands());
    @Nonnull ExplodedConvolutionGrid grid = convolutionLayer
        .getExplodedNetwork();
    convolutionLayer.freeRef();
    tensor.freeRef();
    grid.add(network.getHead());
    grid.freeRef();
    network.wrap(new ReshapeLayer(outputDims)).freeRef();
    network.setName(getName());
    return network;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    @Nullable Tensor tensor = getWeights();
    json.add("weights", tensor.getJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public FullyConnectedLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @Nullable
  public Tensor getWeights() {
    return weights;
  }

  @Nonnull
  public FullyConnectedLayer setWeights(@Nonnull final DoubleSupplier f) {
    Arrays.parallelSetAll(getWeights().getData(), i -> f.getAsDouble());
    return this;
  }

  public int getBatchBands() {
    return batchBands;
  }

  @Nonnull
  public FullyConnectedLayer setBatchBands(int batchBands) {
    this.batchBands = batchBands;
    return this;
  }

  public FullyConnectedLayer set(DoubleSupplier fn) {
    weights.set(fn);
    return this;
  }
}
