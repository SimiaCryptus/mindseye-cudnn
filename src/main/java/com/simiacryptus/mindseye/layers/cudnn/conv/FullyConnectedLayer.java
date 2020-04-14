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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.function.DoubleSupplier;

/**
 * The type Fully connected layer.
 */
@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase implements MultiPrecision, Explodable {
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  /**
   * The Input dims.
   */
  @Nullable
  public final int[] inputDims;
  /**
   * The Output dims.
   */
  @Nullable
  public final int[] outputDims;
  @Nullable
  private final Tensor weights;

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private int batchBands = 0;

  private FullyConnectedLayer() {
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  /**
   * Instantiates a new Fully connected layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outs);
    setWeights(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  /**
   * Instantiates a new Fully connected layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), rs);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  /**
   * Gets batch bands.
   *
   * @return the batch bands
   */
  public int getBatchBands() {
    return batchBands;
  }

  /**
   * Sets batch bands.
   *
   * @param batchBands the batch bands
   */
  public void setBatchBands(int batchBands) {
    this.batchBands = batchBands;
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    assert outputDims != null;
    assert inputDims != null;
    FullyConnectedReferenceLayer referenceLayer = new FullyConnectedReferenceLayer(inputDims, outputDims);
    referenceLayer.set(getWeights());
    return referenceLayer;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * Gets weights.
   *
   * @return the weights
   */
  @Nullable
  public Tensor getWeights() {
    return weights == null ? null : weights.addRef();
  }

  /**
   * Sets weights.
   *
   * @param f the f
   */
  public void setWeights(@Nonnull final DoubleSupplier f) {
    weights.set(i -> f.getAsDouble());
  }

  /**
   * Sets weights log.
   *
   * @param value the value
   */
  public void setWeightsLog(double value) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.setByCoord(c -> (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    weights.freeRef();
  }

  /**
   * From json fully connected layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the fully connected layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  /**
   * Set.
   *
   * @param data the data
   */
  public void set(double[] data) {
    assert weights != null;
    weights.set(data);
  }

  /**
   * Set.
   *
   * @param data the data
   */
  public void set(@Nonnull Tensor data) {
    assert weights != null;
    weights.set(data);
  }

  @Nullable
  @Override
  public Result eval(@Nullable final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    Layer explode = explode();
    Result result = explode.eval(inObj);
    explode.freeRef();
    return result;
  }

  @Nonnull
  public Layer explode() {
    assert inputDims != null;
    int inputVol = Tensor.length(inputDims);
    assert outputDims != null;
    int outVol = Tensor.length(outputDims);
    @Nonnull
    PipelineNetwork network = new PipelineNetwork(1);
    RefUtil.freeRef(network.add(new ReshapeLayer(1, 1, inputVol)));
    assert this.weights != null;
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, inputVol, outVol);
    convolutionLayer.set(this.weights.reshapeCast(1, 1, inputVol * outVol));
    convolutionLayer.setBatchBands(getBatchBands());
    @Nonnull
    ExplodedConvolutionGrid grid = convolutionLayer.getExplodedNetwork();
    convolutionLayer.freeRef();
    grid.add(network.getHead(), network.addRef());
    grid.freeRef();
    RefUtil.freeRef(network.add(new ReshapeLayer(outputDims)));
    network.setName(getName());
    return network;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert outputDims != null;
    json.add("outputDims", JsonUtil.getJson(outputDims));
    assert inputDims != null;
    json.add("inputDims", JsonUtil.getJson(inputDims));
    @Nullable
    Tensor tensor = getWeights();
    assert tensor != null;
    json.add("weights", tensor.getJson(resources, dataSerializer));
    tensor.freeRef();
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(weights.getData());
  }

  /**
   * Set.
   *
   * @param fn the fn
   */
  public void set(@Nonnull DoubleSupplier fn) {
    assert weights != null;
    weights.set(fn);
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  FullyConnectedLayer addRef() {
    return (FullyConnectedLayer) super.addRef();
  }
}
