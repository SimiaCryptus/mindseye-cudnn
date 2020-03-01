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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.ImgCropLayer.Alignment;
import com.simiacryptus.mindseye.layers.java.AvgPoolingLayer;
import com.simiacryptus.mindseye.layers.java.MaxPoolingLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnPoolingMode;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class PoolingLayer extends LayerBase implements MultiPrecision {

  private PoolingMode mode = PoolingMode.Max;
  private int paddingX = 0;
  private int paddingY = 0;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private int strideX = 2;
  private int strideY = 2;
  private int windowX = 2;
  private int windowY = 2;
  private double alpha = 1.0;

  public PoolingLayer() {
    super();
  }

  public PoolingLayer(UUID id, String name) {
    super(id, name);
  }

  protected PoolingLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = RefUtil.get(RefArrays.stream(PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst());
    alpha = json.get("alpha").getAsDouble();
    windowX = json.get("windowX").getAsInt();
    windowY = json.get("windowY").getAsInt();
    paddingX = json.get("paddingX").getAsInt();
    paddingY = json.get("paddingY").getAsInt();
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == PoolingMode.Max)
      return this.as(MaxPoolingLayer.class);
    if (mode == PoolingMode.Avg)
      return this.as(AvgPoolingLayer.class);
    else
      throw new RuntimeException("Not Implemented");
  }

  public PoolingMode getMode() {
    return mode;
  }

  public void setMode(PoolingMode mode) {
    this.mode = mode;
  }

  @Nullable
  @Override
  public String getName() {
    return RefString.format("%sPooling [%d/%d x %d/%d]", mode.name(), windowX, windowY, strideX, strideY);
  }

  public int getPaddingX() {
    return paddingX;
  }

  public void setPaddingX(int paddingX) {
    this.paddingX = paddingX;
  }

  public int getPaddingY() {
    return paddingY;
  }

  public void setPaddingY(int paddingY) {
    this.paddingY = paddingY;
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

  public int getWindowX() {
    return windowX;
  }

  public void setWindowX(int windowX) {
    this.windowX = windowX;
  }

  public int getWindowY() {
    return windowY;
  }

  public void setWindowY(int windowY) {
    this.windowY = windowY;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static PoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PoolingLayer(json);
  }

  @Nonnull
  public static PoolingLayer getPoolingLayer(int radius, PoolingLayer.PoolingMode mode, String qualifier) {
    String name = RefString.format("%s.pool:%s;%s", qualifier, radius, mode);
    PoolingLayer poolingLayer = new PoolingLayer(UUID.nameUUIDFromBytes(name.getBytes()), name);
    poolingLayer.setMode(mode);
    poolingLayer.setStrideXY(radius, radius);
    poolingLayer.setWindowXY(radius, radius);
    return poolingLayer;
  }

  private static int correct(int dim, int modulus, int offset) {
    if (0 >= modulus)
      throw new IllegalArgumentException();
    int lastV = 0;
    while (lastV < dim)
      lastV += modulus;
    lastV -= modulus;
    lastV += offset;
    return lastV - dim;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    TensorList data0 = inObj[0].getData();
    @Nonnull final int[] rawInputDims = data0.getDimensions();
    data0.freeRef();
    int correctionX = correct(rawInputDims[0], strideX, windowX);
    int correctionY = correct(rawInputDims[1], strideY, windowY);
    int paddingX = Math.max(0, PoolingLayer.this.paddingX - (correctionX + 1) / 2);
    int paddingY = Math.max(0, PoolingLayer.this.paddingY - (correctionY + 1) / 2);
    assert paddingX >= 0;
    assert paddingY >= 0;
    assert correctionX >= 0;
    assert correctionY >= 0;
    @Nullable
    Result input = null;
    if (correctionX > 0 || correctionY > 0) {
      ImgPaddingLayer imgPaddingLayer = new ImgPaddingLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY);
      imgPaddingLayer.setPrecision(precision);
      imgPaddingLayer.setHorizontalAlign(Alignment.Center);
      imgPaddingLayer.setVerticalAlign(Alignment.Center);
      imgPaddingLayer.setRoundUp(false);
      RefUtil.freeRef(input);
      input = imgPaddingLayer.eval(inObj[0].addRef());
      imgPaddingLayer.freeRef();
    } else {
      RefUtil.freeRef(input);
      input = inObj[0].addRef();
    }
    assert input != null;
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    final int inputLength = inputData.length();

    final int poolDims = 2;
    @Nonnull final int windowSize[] = {windowY, windowX};
    @Nonnull final int padding[] = {paddingY, paddingX};
    @Nonnull final int stride[] = {strideY, strideX};
    @Nonnull final int[] outputSize = new int[4];
    final CudaTensor outputData = fwd(inputData.addRef(), poolDims, windowSize, padding, stride, outputSize, inObj);
    assert inputDims[2] == outputSize[1];
    final boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(input, outputData.addRef(), inputData, inputLength, alive, inputDims, poolDims, windowSize, padding, stride, precision, mode, alpha);
    CudaTensorList data = new CudaTensorList(outputData, inputLength,
        new int[]{outputSize[3], outputSize[2], outputSize[1]}, precision);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("windowX", windowX);
    json.addProperty("windowY", windowY);
    json.addProperty("paddingX", paddingX);
    json.addProperty("paddingY", paddingY);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("alpha", alpha);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public void setWindowXY(int x, int y) {
    setWindowY(y);
    setWindowX(x);
  }

  public void setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
  }

  public void setPaddingXY(int x, int y) {
    setPaddingX(x);
    setPaddingY(y);
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PoolingLayer addRef() {
    return (PoolingLayer) super.addRef();
  }

  @NotNull
  private CudaTensor fwd(TensorList inputData, int poolDims, int[] windowSize, int[] padding, int[] stride, int[] outputSize, @Nonnull Result[] inObj) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
          try {
            gpu.initThread();
            @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(mode.id, poolDims,
                windowSize, padding, stride);
            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                MemoryType.Device, false);
            CudaSystem.handle(CudaSystem.cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(),
                inputTensor.descriptor.getPtr(), 4, outputSize));
            @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, outputSize[0],
                outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3],
                outputSize[2] * outputSize[3], outputSize[3], 1);
            @Nonnull final CudaMemory outputTensor = gpu.allocate((long) precision.size * Tensor.length(outputSize),
                MemoryType.Managed.ifEnabled(), true);
            CudaMemory inputDataMemory = inputTensor.getMemory(gpu.addRef());
            assert inputDataMemory != null;
            CudaSystem.handle(
                gpu.cudnnPoolingForward(poolingDesc.getPtr(), precision.getPointer(alpha), inputTensor.descriptor.getPtr(),
                    inputDataMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputTensor.getPtr()));
            inputTensor.freeRef();
            poolingDesc.freeRef();
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            gpu.freeRef();
            inputDataMemory.dirty();
            inputDataMemory.freeRef();
            outputTensor.dirty();
            return new CudaTensor(outputTensor,
                outputDescriptor, precision);
          } catch (@Nonnull final Throwable e) {
            throw new ComponentException("Error processing " + RefArrays.stream(RefUtil.addRefs(inObj)).map(x -> {
              TensorList temp_37_0018 = x.getData();
              String temp_37_0004 = RefArrays.toString(temp_37_0018.getDimensions());
              temp_37_0018.freeRef();
              x.freeRef();
              return temp_37_0004;
            }).reduce((a, b1) -> a + ";" + b1) + " with " + this.toString(), e);
          }
        }, inputData.addRef(), inObj),
        inputData);
  }

  public enum PoolingMode {
    Avg(cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING), Max(cudnnPoolingMode.CUDNN_POOLING_MAX);

    final int id;

    PoolingMode(final int id) {
      this.id = id;
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final Result input;
    private final CudaTensor outputData;
    private final TensorList inputData;
    private final int inputLength;
    private final boolean finalInputAlive;
    private final int[] inputDims;
    private final int poolDims;
    private final int[] windowSize;
    private final int[] padding;
    private final int[] stride;
    private final PoolingMode mode;
    private final double alpha;
    private Precision precision;

    public Accumulator(Result input, CudaTensor outputData, TensorList inputData, int inputLength, boolean finalInputAlive, int[] inputDims, int poolDims, int[] windowSize, int[] padding, int[] stride, Precision precision, PoolingMode mode, double alpha) {
      this.input = input;
      this.outputData = outputData;
      this.inputData = inputData;
      this.inputLength = inputLength;
      this.finalInputAlive = finalInputAlive;
      this.inputDims = inputDims;
      this.poolDims = poolDims;
      this.windowSize = windowSize;
      this.padding = padding;
      this.stride = stride;
      this.precision = precision;
      this.mode = mode;
      this.alpha = alpha;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      assert error.length() == inputLength;
      if (finalInputAlive) {
        TensorList data = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                      inputLength, inputDims[2], inputDims[1], inputDims[0],
                      inputDims[2] * inputDims[1] * inputDims[0], inputDims[1] * inputDims[0], inputDims[0], 1);
                  @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(mode.id,
                      poolDims, windowSize, padding, stride);
                  @Nullable CudaTensor inputTensor = null;
                  synchronized (gpu) {
                    RefUtil.freeRef(inputTensor);
                    inputTensor = gpu.getTensor(inputData.addRef(), precision,
                        MemoryType.Device, true);
                  }
                  @Nullable CudaTensor errorPtr = null;
                  synchronized (gpu) {
                    RefUtil.freeRef(errorPtr);
                    errorPtr = gpu.getTensor(error.addRef(), precision,
                        MemoryType.Device, true);
                  }
                  @Nonnull final CudaMemory passbackBuffer = gpu.allocate(
                      (long) Tensor.length(inputDims) * precision.size * inputLength,
                      MemoryType.Managed.ifEnabled(), true);
                  assert outputData != null;
                  CudaMemory outputDataMemory = outputData.getMemory(gpu.addRef());
                  CudaMemory errorPtrMemory = errorPtr.getMemory(gpu.addRef());
                  CudaMemory inputDataMemory = inputTensor.getMemory(gpu.addRef());
                  assert inputDataMemory != null;
                  assert errorPtrMemory != null;
                  assert outputDataMemory != null;
                  CudaSystem.handle(gpu.cudnnPoolingBackward(poolingDesc.getPtr(), precision.getPointer(alpha),
                      outputData.descriptor.getPtr(), outputDataMemory.getPtr(), errorPtr.descriptor.getPtr(),
                      errorPtrMemory.getPtr(), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                      precision.getPointer(0.0), passbackDescriptor.getPtr(), passbackBuffer.getPtr()));
                  gpu.freeRef();
                  errorPtr.freeRef();
                  inputTensor.freeRef();
                  poolingDesc.freeRef();
                  outputDataMemory.dirty();
                  outputDataMemory.freeRef();
                  errorPtrMemory.dirty();
                  errorPtrMemory.freeRef();
                  inputDataMemory.dirty();
                  inputDataMemory.freeRef();
                  passbackBuffer.dirty();
                  return new CudaTensorList(
                      new CudaTensor(passbackBuffer, passbackDescriptor, precision),
                      inputLength, inputDims, precision);
                }, error.addRef(), outputData == null ? null : outputData.addRef(),
                inputData.addRef()), error.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        Result.Accumulator accumulator = input.getAccumulator();
        try {
          accumulator.accept(buffer1, data);
        } finally {
          accumulator.freeRef();
        }
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      input.freeRef();
      outputData.freeRef();
      inputData.freeRef();
    }
  }
}
