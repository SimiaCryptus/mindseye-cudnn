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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class ConvolutionLayer extends LayerBase implements MultiPrecision, Explodable {

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
    int[] kernelDimensions = getKernelDimensions();
    if (kernelDimensions.length != 3)
      throw new IllegalArgumentException();
    if (kernelDimensions[0] <= 0)
      throw new IllegalArgumentException();
    if (kernelDimensions[1] <= 0)
      throw new IllegalArgumentException();
    if (kernelDimensions[2] <= 0)
      throw new IllegalArgumentException();
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    setBatchBands((int) Math.sqrt(CudaSettings.INSTANCE().getMaxFilterElements() / (width * height)));
  }

  protected ConvolutionLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.kernel = Tensor.fromJson(json.get("filter"), resources);
    assert kernel != null;
    assert kernel.isValid();
    assert kernel.rms() > 0;
    setBatchBands(json.get("batchBands").getAsInt());
    setStrideX(json.get("strideX").getAsInt());
    setStrideY(json.get("strideY").getAsInt());
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive()) {
      setPaddingX(paddingX.getAsInt());
    }
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive()) {
      setPaddingY(paddingY.getAsInt());
    }
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    this.inputBands = json.get("inputBands").getAsInt();
    this.outputBands = json.get("outputBands").getAsInt();
  }

  public int getBatchBands() {
    return batchBands;
  }

  public void setBatchBands(int batchBands) {
    this.batchBands = batchBands;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
    //    return this.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }

  @Nonnull
  public ConvolutionParams getConvolutionParams() {
    return new ConvolutionParams(inputBands, outputBands, precision, strideX, strideY, paddingX, paddingY, getKernelDimensions());
  }

  @Nonnull
  public ExplodedConvolutionGrid getExplodedNetwork() {
    assertAlive();
    int batchBands = getBatchBands();
    if (0 == batchBands) {
      batchBands = Math.max(inputBands, outputBands);
    }
    ExplodedConvolutionGrid grid = new ExplodedConvolutionGrid(getConvolutionParams(), batchBands);
    grid.write(getKernel());
    //    if (batchBands > outputBands * 2) {
    //      batchBands = outputBands;
    //    }
    return grid;
  }

  @Nullable
  public Tensor getKernel() {
    return kernel.addRef();
  }

  public int[] getKernelDimensions() {
    assert kernel != null;
    return kernel.getDimensions();
  }

  @Nullable
  @Override
  public String getName() {
    int[] kernelDimensions = getKernelDimensions();
    if (kernelDimensions.length == 4) {
      return RefString.format("Conv [%d/%d x %d/%d, %d -> %d]", kernelDimensions[0], strideX, kernelDimensions[1],
          strideY, kernelDimensions[2], kernelDimensions[3]);
    } else {
      return RefString.format("Conv [%d/%d x %d/%d, %d]", kernelDimensions[0], strideX, kernelDimensions[1], strideY,
          kernelDimensions[2]);
    }
  }

  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }

  public void setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
  }

  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  public void setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public int getStrideX() {
    return strideX;
  }

  public void setStrideX(int strideX) {
    this.strideX = strideX;
  }

  public int getStrideY() {
    return strideY;
  }

  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }

  public void setByCoord(ToDoubleFunction<Coordinate> coordinateToDoubleFunction) {
    kernel.setByCoord(coordinateToDoubleFunction);
    assert kernel.rms() > 0;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }

  @Nonnull
  @Override
  public Layer explode() {
    @Nonnull
    ExplodedConvolutionGrid explodedNetwork = getExplodedNetwork();
    @Nonnull
    Layer network = explodedNetwork.getNetwork();
    network.setName(getName() + "+");
    explodedNetwork.freeRef();
    return network;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Tensor kernel = getKernel();
    assert kernel != null;
    assert kernel.isValid();
    assert 1 == inObj.length;
    TensorList data0 = inObj[0].getData();
    int[] data0Dimensions = data0.getDimensions();
    assert 3 == data0Dimensions.length;
    assert inputBands == data0Dimensions[2] : RefArrays.toString(data0Dimensions) + "[2] != " + inputBands;
    if (!CudaSystem.isEnabled()) {
      kernel.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      data0.freeRef();
      return result;
    }
    @Nonnull
    ExplodedConvolutionGrid grid = getExplodedNetwork();
    @Nonnull
    PipelineNetwork network = grid.getNetwork();
    if (isFrozen()) {
      network.freeze();
    }
    final Result result = network.eval(RefUtil.addRef(inObj));
    network.freeRef();
    assert result != null;
    final TensorList resultData = result.getData();
    assert data0.length() == resultData.length();
    assert 3 == resultData.getDimensions().length;
    assert outputBands == resultData.getDimensions()[2];
    data0.freeRef();
    RefUtil.freeRef(inObj);
    boolean alive = result.isAlive();
    Result.Accumulator accumulator = new Accumulator(kernel, grid, isFrozen(), result.getAccumulator(), getId());
    result.freeRef();
    return new Result(resultData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.getJson(resources, dataSerializer));
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

  public void set(@Nonnull DoubleSupplier f) {
    set(i -> f.getAsDouble());
  }

  public void set(@Nonnull Tensor tensor) {
    kernel.set(tensor);
    assert kernel.rms() > 0;
  }

  public void set(@Nonnull IntToDoubleFunction f) {
    kernel.set(f);
    assert kernel.rms() > 0;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  public void setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
  }

  public void setPaddingXY(Integer x, Integer y) {
    setPaddingX(x);
    setPaddingY(y);
  }

  public void _free() {
    if (null != kernel)
      kernel.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ConvolutionLayer addRef() {
    return (ConvolutionLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor kernel;
    private final ExplodedConvolutionGrid grid;
    private boolean frozen;
    private Result.Accumulator accumulator;
    private UUID id;

    public Accumulator(Tensor kernel, ExplodedConvolutionGrid grid, boolean frozen, Result.Accumulator accumulator, UUID id) {
      this.kernel = kernel;
      this.grid = grid;
      this.frozen = frozen;
      this.accumulator = accumulator;
      this.id = id;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaSet, @Nullable TensorList delta) {
      assert deltaSet != null;
      this.accumulator.accept(deltaSet.addRef(), delta);
      if (!frozen) {
        Tensor read = grid.read(deltaSet.addRef(), true);
        Delta<UUID> uuidDelta = deltaSet.get(id, kernel.addRef());
        assert uuidDelta != null;
        uuidDelta.addInPlace(read);
        uuidDelta.freeRef();
      }
      if (null != deltaSet)
        deltaSet.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      kernel.freeRef();
      grid.freeRef();
      accumulator.freeRef();
    }
  }
}
