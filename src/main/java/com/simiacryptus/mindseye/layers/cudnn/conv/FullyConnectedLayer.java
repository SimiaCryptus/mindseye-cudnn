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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.function.DoubleSupplier;

@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase implements MultiPrecision, Explodable {
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
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    Tensor temp_15_0002 = new Tensor(inputs, outs);
    weights = temp_15_0002.addRef();
    temp_15_0002.freeRef();
    setWeights(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    @Nullable final Tensor data = Tensor.fromJson(json.get("weights"), rs);
    Tensor temp_15_0003 = data == null ? null : data.addRef();
    weights = temp_15_0003 == null ? null : temp_15_0003.addRef();
    if (null != temp_15_0003)
      temp_15_0003.freeRef();
    if (null != data)
      data.freeRef();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  public int getBatchBands() {
    return batchBands;
  }

  public void setBatchBands(int batchBands) {
    this.batchBands = batchBands;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    assert outputDims != null;
    assert inputDims != null;
    FullyConnectedReferenceLayer temp_15_0007 = new FullyConnectedReferenceLayer(inputDims, outputDims);
    temp_15_0007.set(getWeights());
    FullyConnectedReferenceLayer temp_15_0006 = temp_15_0007.addRef();
    temp_15_0007.freeRef();
    return temp_15_0006;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nullable
  public Tensor getWeights() {
    return weights == null ? null : weights.addRef();
  }

  @Nonnull
  public void setWeights(@Nonnull final DoubleSupplier f) {
    Tensor temp_15_0009 = getWeights();
    assert temp_15_0009 != null;
    RefArrays.parallelSetAll(temp_15_0009.getData(), i -> f.getAsDouble());
    temp_15_0009.freeRef();
  }

  public void setWeightsLog(double value) {
    Tensor temp_15_0010 = getWeights();
    assert temp_15_0010 != null;
    temp_15_0010.setByCoord(c -> (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    temp_15_0010.freeRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  public void set(double[] data) {
    assert weights != null;
    weights.set(data);
  }

  public void set(@Nonnull Tensor data) {
    assert weights != null;
    weights.set(data);
  }

  @Nullable
  @Override
  public Result eval(@Nullable final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_15_0011 = getCompatibilityLayer();
      Result temp_15_0005 = temp_15_0011.eval(RefUtil.addRefs(inObj));
      temp_15_0011.freeRef();
      if (null != inObj)
        ReferenceCounting.freeRefs(inObj);
      return temp_15_0005;
    }
    Layer explode = explode();
    Result temp_15_0004 = explode.eval(RefUtil.addRefs(inObj));
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    explode.freeRef();
    return temp_15_0004;
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
    @Nullable
    Tensor tensor = this.weights.reshapeCast(1, 1, inputVol * outVol);
    ConvolutionLayer temp_15_0008 = new ConvolutionLayer(1, 1, inputVol, outVol);
    temp_15_0008.set(tensor.addRef());
    ConvolutionLayer temp_15_0012 = temp_15_0008.addRef();
    temp_15_0012.setBatchBands(getBatchBands());
    @Nonnull
    ConvolutionLayer convolutionLayer = temp_15_0012.addRef();
    temp_15_0012.freeRef();
    temp_15_0008.freeRef();
    tensor.freeRef();
    @Nonnull
    ExplodedConvolutionGrid grid = convolutionLayer.getExplodedNetwork();
    convolutionLayer.freeRef();
    grid.add(network.getHead());
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
    Tensor temp_15_0014 = getWeights();
    assert temp_15_0014 != null;
    RefList<double[]> temp_15_0013 = RefArrays.asList(temp_15_0014.getData());
    temp_15_0014.freeRef();
    return temp_15_0013;
  }

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
