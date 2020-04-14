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

/**
 * The type Convolution layer.
 */
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
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private int batchBands = 0;

  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this(1, 1, 1, 1);
  }

  /**
   * Instantiates a new Convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
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
    setBatchBands((int) Math.sqrt(CudaSettings.INSTANCE().maxFilterElements / (width * height)));
  }

  /**
   * Instantiates a new Convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
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
    return null;
    //    return this.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }

  /**
   * Gets convolution params.
   *
   * @return the convolution params
   */
  @Nonnull
  public ConvolutionParams getConvolutionParams() {
    return new ConvolutionParams(inputBands, outputBands, precision, strideX, strideY, paddingX, paddingY, getKernelDimensions());
  }

  /**
   * Gets exploded network.
   *
   * @return the exploded network
   */
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

  /**
   * Gets kernel.
   *
   * @return the kernel
   */
  @Nullable
  public Tensor getKernel() {
    return kernel.addRef();
  }

  /**
   * Get kernel dimensions int [ ].
   *
   * @return the int [ ]
   */
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

  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }

  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   */
  public void setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
  }

  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   */
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

  /**
   * Gets stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }

  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   */
  public void setStrideX(int strideX) {
    this.strideX = strideX;
  }

  /**
   * Gets stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }

  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   */
  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }

  /**
   * Sets by coord.
   *
   * @param coordinateToDoubleFunction the coordinate to double function
   */
  public void setByCoord(ToDoubleFunction<Coordinate> coordinateToDoubleFunction) {
    kernel.setByCoord(coordinateToDoubleFunction);
    assert kernel.rms() > 0;
  }

  /**
   * From json convolution layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the convolution layer
   */
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

  /**
   * Set.
   *
   * @param f the f
   */
  public void set(@Nonnull DoubleSupplier f) {
    set(i -> f.getAsDouble());
  }

  /**
   * Set.
   *
   * @param tensor the tensor
   */
  public void set(@Nonnull Tensor tensor) {
    kernel.set(tensor);
    assert kernel.rms() > 0;
  }

  /**
   * Set.
   *
   * @param f the f
   */
  public void set(@Nonnull IntToDoubleFunction f) {
    kernel.set(f);
    assert kernel.rms() > 0;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  /**
   * Sets stride xy.
   *
   * @param x the x
   * @param y the y
   */
  public void setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
  }

  /**
   * Sets padding xy.
   *
   * @param x the x
   * @param y the y
   */
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

    /**
     * Instantiates a new Accumulator.
     *
     * @param kernel      the kernel
     * @param grid        the grid
     * @param frozen      the frozen
     * @param accumulator the accumulator
     * @param id          the id
     */
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
