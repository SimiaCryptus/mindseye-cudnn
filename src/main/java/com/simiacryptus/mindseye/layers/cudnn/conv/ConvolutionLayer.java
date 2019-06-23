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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ConvolutionLayer extends LayerBase implements MultiPrecision<ConvolutionLayer>, Explodable {

  @Nullable
  private final Tensor kernel;
  private final int inputBands;
  private final int outputBands;
  private int strideX = 1;
  private int strideY = 1;
  @Nullable
  private Integer paddingX = null;
  @Nullable
  private Integer paddingY = null;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private int batchBands = 0;

  protected ConvolutionLayer() {
    this(1, 1, 1, 1);
  }

  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands) {
    super();
    assert 0 < width;
    assert 0 < height;
    assert 0 < inputBands;
    assert 0 < outputBands;
    this.kernel = new Tensor(width, height, inputBands * outputBands);
    if (getKernel().getDimensions().length != 3) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    int batchBands = (int) Math.sqrt(CudaSettings.INSTANCE().getMaxFilterElements() / (width * height));
    //batchBands = binaryFriendly(batchBands, 3);
    setBatchBands(batchBands);
  }

  protected ConvolutionLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.kernel = Tensor.fromJson(json.get("filter"), resources);
    assert getKernel().isValid();
    this.setBatchBands(json.get("batchBands").getAsInt());
    this.setStrideX(json.get("strideX").getAsInt());
    this.setStrideY(json.get("strideY").getAsInt());
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive()) this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive()) this.setPaddingY((paddingY.getAsInt()));
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    this.inputBands = json.get("inputBands").getAsInt();
    this.outputBands = json.get("outputBands").getAsInt();
  }

  public static int binaryFriendly(final int value, final int bits) {
    return (int) Math.pow(2, (Math.floor(Math.log(value) * bits) / bits) / Math.log(2));
  }

  public static void add(@Nonnull final DoubleSupplier f, @Nonnull final double[] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] += f.getAsDouble();
    }
  }

  public static ConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }

  @Nullable
  @Override
  public String getName() {
    int[] kernelDimensions = kernel.getDimensions();
    if (kernelDimensions.length == 4) {
      return String.format("Conv [%d/%d x %d/%d, %d -> %d]", kernelDimensions[0], strideX, kernelDimensions[1], strideY, kernelDimensions[2], kernelDimensions[3]);
    } else {
      return String.format("Conv [%d/%d x %d/%d, %d]", kernelDimensions[0], strideX, kernelDimensions[1], strideY, kernelDimensions[2]);
    }
  }

  @Override
  protected void _free() {
    kernel.freeRef();
    super._free();
  }

  @Nonnull
  public ConvolutionLayer addWeights(@Nonnull final DoubleSupplier f) {
    ConvolutionLayer.add(f, getKernel().getData());
    return this;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
//    return this.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }

  public Layer explodeAndFree() {
    Layer explode = explode();
    freeRef();
    return explode;
  }


  @Nonnull
  @Override
  public Layer explode() {
    @Nonnull ExplodedConvolutionGrid explodedNetwork = getExplodedNetwork();
    try {
      @Nonnull Layer network = explodedNetwork.getNetwork();
      network.setName(getName() + "+");
      return network;
    } finally {
      explodedNetwork.freeRef();
    }
  }

  @Nonnull
  public ExplodedConvolutionGrid getExplodedNetwork() {
    assertAlive();
    int batchBands = getBatchBands();
    if (0 == batchBands) {
      batchBands = Math.max(inputBands, outputBands);
    }
//    if (batchBands > outputBands * 2) {
//      batchBands = outputBands;
//    }
    return new ExplodedConvolutionGrid(getConvolutionParams(), batchBands).write(kernel);
  }

  @Nonnull
  public ConvolutionParams getConvolutionParams() {
    return new ConvolutionParams(inputBands, outputBands, precision, strideX, strideY, paddingX, paddingY, kernel.getDimensions());
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    final Tensor kernel = getKernel();
    kernel.addRef();
    assert kernel.isValid();
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    assert inputBands == inObj[0].getData().getDimensions()[2] : Arrays.toString(inObj[0].getData().getDimensions()) + "[2] != " + inputBands;
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    @Nonnull ExplodedConvolutionGrid grid = getExplodedNetwork();
    @Nonnull PipelineNetwork network = grid.getNetwork();
    final Result result;
    try {
      if (isFrozen()) {
        network.freeze();
      }
      result = network.evalAndFree(inObj);
    } finally {
      network.freeRef();
    }
    final TensorList resultData = result.getData();
    assert inObj[0].getData().length() == resultData.length();
    assert 3 == resultData.getDimensions().length;
    assert outputBands == resultData.getDimensions()[2];
    ConvolutionLayer.this.addRef();
    return new Result(resultData, (@Nonnull final DeltaSet<UUID> deltaSet, @Nonnull final TensorList delta) -> {
      result.accumulate(deltaSet, delta);
      if (!isFrozen()) {
        Tensor read = grid.read(deltaSet, true);
        deltaSet.get(ConvolutionLayer.this.getId(), kernel.getData()).addInPlace(read.getData()).freeRef();
        read.freeRef();
      }
    }) {

      @Override
      public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      @Override
      protected void _free() {
        grid.freeRef();
        result.freeRef();
        kernel.freeRef();
        ConvolutionLayer.this.freeRef();
      }

      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("filter", getKernel().getJson(resources, dataSerializer));
    json.addProperty("batchBands", getBatchBands());
    json.addProperty("strideX", getStrideX());
    json.addProperty("strideY", getStrideY());
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("precision", precision.name());
    json.addProperty("inputBands", inputBands);
    json.addProperty("outputBands", outputBands);
    return json;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ConvolutionLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @Nonnull
  public ConvolutionLayer set(@Nonnull final DoubleSupplier f) {
    return set(i -> f.getAsDouble());
  }

  @Nonnull
  public ConvolutionLayer set(@Nonnull final Tensor tensor) {
    getKernel().set(tensor);
    return this;
  }

  @Nonnull
  public ConvolutionLayer setAndFree(@Nonnull final Tensor tensor) {
    set(tensor);
    tensor.freeRef();
    return this;
  }

  @Nonnull
  public ConvolutionLayer set(@Nonnull final IntToDoubleFunction f) {
    getKernel().set(f);
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(getKernel().getData());
  }

  public int getStrideX() {
    return strideX;
  }

  @Nonnull
  public ConvolutionLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this;
  }

  public int getStrideY() {
    return strideY;
  }

  @Nonnull
  public ConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }

  @Nonnull
  public ConvolutionLayer setWeightsLog(double f) {
    return set(() -> Math.pow(10, f) * (Math.random() - 0.5));
  }

  @Nonnull
  public ConvolutionLayer setStrideXY(int x, int y) {
    return setStrideX(x).setStrideY(y);
  }

  @Nonnull
  public ConvolutionLayer setPaddingXY(Integer x, Integer y) {
    return setPaddingX(x).setPaddingY(y);
  }

  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }

  @Nonnull
  public ConvolutionLayer setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
    return this;
  }

  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public ConvolutionLayer setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
    return this;
  }

  @Nullable
  public Tensor getKernel() {
    return kernel;
  }


  public int getBatchBands() {
    return batchBands;
  }

  @Nonnull
  public ConvolutionLayer setBatchBands(int batchBands) {
    this.batchBands = batchBands;
    return this;
  }
}
