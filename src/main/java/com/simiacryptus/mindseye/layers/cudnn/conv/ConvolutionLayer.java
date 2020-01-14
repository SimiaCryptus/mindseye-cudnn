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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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
    Tensor temp_04_0001 = new Tensor(width, height, inputBands * outputBands);
    this.kernel = temp_04_0001.addRef();
    temp_04_0001.freeRef();
    Tensor temp_04_0011 = getKernel();
    assert temp_04_0011 != null;
    if (temp_04_0011.getDimensions().length != 3)
      throw new IllegalArgumentException();
    temp_04_0011.freeRef();
    Tensor temp_04_0012 = getKernel();
    if (temp_04_0012.getDimensions()[0] <= 0)
      throw new IllegalArgumentException();
    temp_04_0012.freeRef();
    Tensor temp_04_0013 = getKernel();
    if (temp_04_0013.getDimensions()[1] <= 0)
      throw new IllegalArgumentException();
    temp_04_0013.freeRef();
    Tensor temp_04_0014 = getKernel();
    if (temp_04_0014.getDimensions()[2] <= 0)
      throw new IllegalArgumentException();
    temp_04_0014.freeRef();
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    int batchBands = (int) Math.sqrt(CudaSettings.INSTANCE().getMaxFilterElements() / (width * height));
    RefUtil.freeRef(setBatchBands(batchBands));
  }

  protected ConvolutionLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor temp_04_0002 = Tensor.fromJson(json.get("filter"), resources);
    this.kernel = temp_04_0002 == null ? null : temp_04_0002.addRef();
    if (null != temp_04_0002)
      temp_04_0002.freeRef();
    Tensor temp_04_0015 = getKernel();
    assert temp_04_0015 != null;
    assert temp_04_0015.isValid();
    temp_04_0015.freeRef();
    RefUtil.freeRef(this.setBatchBands(json.get("batchBands").getAsInt()));
    RefUtil.freeRef(this.setStrideX(json.get("strideX").getAsInt()));
    RefUtil.freeRef(this.setStrideY(json.get("strideY").getAsInt()));
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive())
      this.setPaddingX((paddingX.getAsInt())).freeRef();
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive())
      this.setPaddingY((paddingY.getAsInt())).freeRef();
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    this.inputBands = json.get("inputBands").getAsInt();
    this.outputBands = json.get("outputBands").getAsInt();
  }

  public int getBatchBands() {
    return batchBands;
  }

  @Nonnull
  public ConvolutionLayer setBatchBands(int batchBands) {
    this.batchBands = batchBands;
    return this.addRef();
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
    //    return this.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }

  @Nonnull
  public ConvolutionParams getConvolutionParams() {
    assert kernel != null;
    return new ConvolutionParams(inputBands, outputBands, precision, strideX, strideY, paddingX, paddingY,
        kernel.getDimensions());
  }

  @Nonnull
  public ExplodedConvolutionGrid getExplodedNetwork() {
    assertAlive();
    int batchBands = getBatchBands();
    if (0 == batchBands) {
      batchBands = Math.max(inputBands, outputBands);
    }
    ExplodedConvolutionGrid temp_04_0010 = new ExplodedConvolutionGrid(getConvolutionParams(), batchBands);
    ExplodedConvolutionGrid temp_04_0009 = temp_04_0010.write(kernel == null ? null : kernel.addRef());
    temp_04_0010.freeRef();
    //    if (batchBands > outputBands * 2) {
    //      batchBands = outputBands;
    //    }
    return temp_04_0009;
  }

  @Nullable
  public Tensor getKernel() {
    return kernel == null ? null : kernel.addRef();
  }

  @Nullable
  @Override
  public String getName() {
    assert kernel != null;
    int[] kernelDimensions = kernel.getDimensions();
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

  @Nonnull
  public ConvolutionLayer setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
    return this.addRef();
  }

  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public ConvolutionLayer setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ConvolutionLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public int getStrideX() {
    return strideX;
  }

  @Nonnull
  public ConvolutionLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this.addRef();
  }

  public int getStrideY() {
    return strideY;
  }

  @Nonnull
  public ConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ConvolutionLayer[] addRefs(@Nullable ConvolutionLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayer::addRef)
        .toArray((x) -> new ConvolutionLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ConvolutionLayer[][] addRefs(@Nullable ConvolutionLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayer::addRefs)
        .toArray((x) -> new ConvolutionLayer[x][]);
  }

  @Nonnull
  @Override
  public Layer explode() {
    @Nonnull
    ExplodedConvolutionGrid explodedNetwork = getExplodedNetwork();
    @Nonnull
    Layer network = explodedNetwork.getNetwork();
    RefUtil.freeRef(network.setName(getName() + "+"));
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
    TensorList temp_04_0016 = inObj[0].getData();
    assert 3 == temp_04_0016.getDimensions().length;
    temp_04_0016.freeRef();
    TensorList temp_04_0017 = inObj[0].getData();
    TensorList temp_04_0018 = inObj[0].getData();
    assert inputBands == temp_04_0017.getDimensions()[2] : RefArrays.toString(temp_04_0018.getDimensions()) + "[2] != "
        + inputBands;
    temp_04_0018.freeRef();
    temp_04_0017.freeRef();
    if (!CudaSystem.isEnabled()) {
      kernel.freeRef();
      Layer temp_04_0019 = getCompatibilityLayer();
      Result temp_04_0008 = temp_04_0019.eval(Result.addRefs(inObj));
      temp_04_0019.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_04_0008;
    }
    @Nonnull
    ExplodedConvolutionGrid grid = getExplodedNetwork();
    @Nonnull
    PipelineNetwork network = grid.getNetwork();
    final Result result;
    if (isFrozen()) {
      RefUtil.freeRef(network.freeze());
    }
    result = network.eval(Result.addRefs(inObj));
    network.freeRef();
    assert result != null;
    final TensorList resultData = result.getData();
    TensorList temp_04_0020 = inObj[0].getData();
    assert temp_04_0020.length() == resultData.length();
    temp_04_0020.freeRef();
    ReferenceCounting.freeRefs(inObj);
    assert 3 == resultData.getDimensions().length;
    assert outputBands == resultData.getDimensions()[2];
    final ConvolutionLayer convolutionLayer = ConvolutionLayer.this.addRef();
    try {
      try {
        try {
          try {
            try {
              return new Result(resultData, new Result.Accumulator() {
                {
                }

                @Override
                public void accept(@Nullable DeltaSet<UUID> deltaSet, @Nullable TensorList delta) {
                  result.accumulate(deltaSet == null ? null : deltaSet.addRef(), delta == null ? null : delta.addRef());
                  if (null != delta)
                    delta.freeRef();
                  if (!ConvolutionLayer.this.isFrozen()) {
                    Tensor read = grid.read(deltaSet == null ? null : deltaSet.addRef(), true);
                    assert deltaSet != null;
                    Delta<UUID> temp_04_0021 = deltaSet.get(convolutionLayer.getId(), kernel.getData());
                    assert temp_04_0021 != null;
                    RefUtil.freeRef(temp_04_0021.addInPlace(read.getData()));
                    temp_04_0021.freeRef();
                    read.freeRef();
                  }
                  if (null != deltaSet)
                    deltaSet.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
                }
              }) {

                {
                }

                @Override
                public boolean isAlive() {
                  return result.isAlive();
                }

                @Override
                public void accumulate(@Nullable final DeltaSet<UUID> buffer, @Nullable final TensorList delta) {
                  Result.Accumulator temp_04_0022 = getAccumulator();
                  assert temp_04_0022 != null;
                  temp_04_0022.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                  temp_04_0022.freeRef();
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public void _free() {
                }
              };
            } finally {
              convolutionLayer.freeRef();
            }
          } finally {
            resultData.freeRef();
          }
        } finally {
          result.freeRef();
        }
      } finally {
        grid.freeRef();
      }
    } finally {
      kernel.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    Tensor temp_04_0023 = getKernel();
    assert temp_04_0023 != null;
    json.add("filter", temp_04_0023.getJson(resources, dataSerializer));
    temp_04_0023.freeRef();
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

  @Nonnull
  public ConvolutionLayer set(@Nonnull final DoubleSupplier f) {
    return set(i -> f.getAsDouble());
  }

  @Nonnull
  public ConvolutionLayer set(@Nonnull final Tensor tensor) {
    Tensor temp_04_0024 = getKernel();
    assert temp_04_0024 != null;
    temp_04_0024.set(tensor);
    temp_04_0024.freeRef();
    return this.addRef();
  }

  @Nonnull
  public ConvolutionLayer set(@Nonnull final IntToDoubleFunction f) {
    Tensor temp_04_0025 = getKernel();
    assert temp_04_0025 != null;
    RefUtil.freeRef(temp_04_0025.set(f));
    temp_04_0025.freeRef();
    return this.addRef();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor temp_04_0027 = getKernel();
    assert temp_04_0027 != null;
    RefList<double[]> temp_04_0026 = RefArrays.asList(temp_04_0027.getData());
    temp_04_0027.freeRef();
    return temp_04_0026;
  }

  @Nonnull
  public ConvolutionLayer setStrideXY(int x, int y) {
    ConvolutionLayer temp_04_0029 = setStrideX(x);
    ConvolutionLayer temp_04_0028 = temp_04_0029.setStrideY(y);
    temp_04_0029.freeRef();
    return temp_04_0028;
  }

  @Nonnull
  public ConvolutionLayer setPaddingXY(Integer x, Integer y) {
    ConvolutionLayer temp_04_0031 = setPaddingX(x);
    ConvolutionLayer temp_04_0030 = temp_04_0031.setPaddingY(y);
    temp_04_0031.freeRef();
    return temp_04_0030;
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
}
