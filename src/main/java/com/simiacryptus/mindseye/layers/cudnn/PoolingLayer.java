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
import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.layers.java.AvgPoolingLayer;
import com.simiacryptus.mindseye.layers.java.MaxPoolingLayer;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnPoolingMode;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

@SuppressWarnings("serial")
public class PoolingLayer extends LayerBase implements MultiPrecision<PoolingLayer> {

  private PoolingMode mode = PoolingMode.Max;
  private int paddingX = 0;
  private int paddingY = 0;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private int strideX = 2;
  private int strideY = 2;
  private int windowX = 2;
  private int windowY = 2;
  private double alpha;

  public PoolingLayer() {
    super();
    alpha = 1.0;
  }

  public PoolingLayer(UUID id, String name) {
    super(id, name);
  }

  protected PoolingLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    alpha = json.get("alpha").getAsDouble();
    windowX = json.get("windowX").getAsInt();
    windowY = json.get("windowY").getAsInt();
    paddingX = json.get("paddingX").getAsInt();
    paddingY = json.get("paddingY").getAsInt();
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = 1.0;
  }

  public static PoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PoolingLayer(json);
  }

  private static int correct(int dim, int modulus, int offset) {

    // modulus * n + offset == r + dim
    // modulus * n + (offset - dim) == r
    if (0 >= modulus) throw new IllegalArgumentException();
    int lastV = 0;
    while (lastV < dim) lastV += modulus;
    lastV -= modulus;
    lastV += offset;
    return lastV - dim;

//    int adj = modulus - ((dim - offset) % modulus);
//    while (adj < 0) adj += modulus;
//    while (adj >= modulus) adj -= modulus;
//    return adj;
  }

  public static PoolingLayer getPoolingLayer(int radius, PoolingLayer.PoolingMode mode) {
    String name = String.format("Conv:%s;%s", radius, mode);
    return new PoolingLayer(UUID.nameUUIDFromBytes(name.getBytes()), name)
        .setMode(mode)
        .setStrideXY(radius, radius)
        .setWindowXY(radius, radius);
  }

  @Nullable
  @Override
  public String getName() {
    return String.format("%sPooling [%d/%d x %d/%d]", mode.name(), windowX, windowY, strideX, strideY);
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == PoolingMode.Max) return this.as(MaxPoolingLayer.class);
    if (mode == PoolingMode.Avg) return this.as(AvgPoolingLayer.class);
    else throw new RuntimeException("Not Implemented");
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    @Nonnull final int[] rawInputDims = inObj[0].getData().getDimensions();

    int correctionX = correct(rawInputDims[0], strideX, windowX);
    int correctionY = correct(rawInputDims[1], strideY, windowY);
    int paddingX = Math.max(0, PoolingLayer.this.paddingX - ((correctionX + 1) / 2));
    int paddingY = Math.max(0, PoolingLayer.this.paddingY - ((correctionY + 1) / 2));
//    if (correctionX >= windowX) correctionX -= windowX;
//    if (correctionY >= windowY) correctionY -= windowY;
    assert paddingX >= 0;
    assert paddingY >= 0;
    assert correctionX >= 0;
    assert correctionY >= 0;
    @Nullable Result input;
    if (correctionX > 0 || correctionY > 0) {
      @Nonnull Layer paddingLayer = new ImgPaddingLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY)
          .setPrecision(precision)
          .setHorizontalAlign(ImgPaddingLayer.Alignment.Center)
          .setVerticalAlign(ImgPaddingLayer.Alignment.Center)
          .setRoundUp(false);
      input = paddingLayer.evalAndFree(inObj[0]);
      paddingLayer.freeRef();
//      return input;
    } else {
      input = inObj[0];
    }
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    final int inputLength = inputData.length();

    final int poolDims = 2;
    @Nonnull final int windowSize[] = {windowY, windowX};
    @Nonnull final int padding[] = {paddingY, paddingX};
    @Nonnull final int stride[] = {strideY, strideX};
    @Nonnull final int[] outputSize = new int[4];
    final CudaTensor outputData = CudaSystem.run(gpu -> {
      try {
        gpu.initThread();
        @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(
            mode.id, poolDims, windowSize, padding, stride);
        @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        CudaSystem.handle(CudaSystem.cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(), inputTensor.descriptor.getPtr(), 4, outputSize));
        assert inputDims[2] == outputSize[1];
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, outputSize[0], outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3], outputSize[2] * outputSize[3], outputSize[3], 1);
        @Nonnull final CudaMemory outputTensor = gpu.allocate((long) precision.size * Tensor.length(outputSize), MemoryType.Managed.ifEnabled(), true);
        CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
        CudaSystem.handle(gpu.cudnnPoolingForward(poolingDesc.getPtr(),
            precision.getPointer(alpha),
            inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
            precision.getPointer(0.0),
            outputDescriptor.getPtr(), outputTensor.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputDataMemory.dirty();
        outputTensor.dirty();
        Stream.<ReferenceCounting>of(inputTensor, poolingDesc, inputDataMemory).forEach(ReferenceCounting::freeRef);
        return CudaTensor.wrap(outputTensor, outputDescriptor, precision);
      } catch (@Nonnull final Throwable e) {
        throw new ComponentException("Error processing " + Arrays.stream(inObj).map(x -> Arrays.toString(x.getData().getDimensions())).reduce((a, b) -> a + ";" + b) + " with " + this.toString(), e);
      }
    }, inputData);
    return new Result(CudaTensorList.create(outputData, inputLength, new int[]{outputSize[3], outputSize[2], outputSize[1]}, precision),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
          assert error.length() == inputLength;
          if (input.isAlive()) {
            TensorList data = CudaSystem.run(gpu -> {
              @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                  inputLength, inputDims[2], inputDims[1], inputDims[0],
                  inputDims[2] * inputDims[1] * inputDims[0], inputDims[1] * inputDims[0], inputDims[0], 1);
              @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(
                  mode.id, poolDims, windowSize, padding, stride);
              @Nullable final CudaTensor inputTensor;
              synchronized (gpu) {
                inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
              }
              @Nullable final CudaTensor errorPtr;
              synchronized (gpu) {
                errorPtr = gpu.getTensor(error, precision, MemoryType.Device, true);
              }
              @Nonnull final CudaMemory passbackBuffer = gpu.allocate((long) Tensor.length(inputDims) * precision.size * inputLength, MemoryType.Managed.ifEnabled(), true);
              CudaMemory outputDataMemory = outputData.getMemory(gpu);
              CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
              CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
              CudaSystem.handle(gpu.cudnnPoolingBackward(poolingDesc.getPtr(),
                  precision.getPointer(this.alpha), outputData.descriptor.getPtr(), outputDataMemory.getPtr(),
                  errorPtr.descriptor.getPtr(), errorPtrMemory.getPtr(),
                  inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                  precision.getPointer(0.0), passbackDescriptor.getPtr(), passbackBuffer.getPtr()));
              outputDataMemory.dirty();
              errorPtrMemory.dirty();
              inputDataMemory.dirty();
              passbackBuffer.dirty();

              Stream.<ReferenceCounting>of(errorPtr, inputTensor, poolingDesc, outputDataMemory, errorPtrMemory, inputDataMemory).forEach(ReferenceCounting::freeRef);
              return CudaTensorList.wrap(CudaTensor.wrap(passbackBuffer, passbackDescriptor, precision), inputLength, inputDims, precision);
            }, error);
            input.accumulate(buffer, data);
          }
          error.freeRef();
        }) {

      @Override
      protected void _free() {
        input.freeRef();
        inputData.freeRef();
        outputData.freeRef();
      }

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
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

  public PoolingMode getMode() {
    return mode;
  }

  @Nonnull
  public PoolingLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this;
  }

  public int getPaddingX() {
    return paddingX;
  }

  @Nonnull
  public PoolingLayer setPaddingX(final int paddingX) {
    this.paddingX = paddingX;
    return this;
  }

  public int getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public PoolingLayer setPaddingY(final int paddingY) {
    this.paddingY = paddingY;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public PoolingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public int getStrideX() {
    return strideX;
  }

  @Nonnull
  public PoolingLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this;
  }

  public int getStrideY() {
    return strideY;
  }

  @Nonnull
  public PoolingLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this;
  }

  public int getWindowX() {
    return windowX;
  }

  @Nonnull
  public PoolingLayer setWindowX(final int windowX) {
    this.windowX = windowX;
    return this;
  }

  public int getWindowY() {
    return windowY;
  }

  @Nonnull
  public PoolingLayer setWindowY(final int windowY) {
    this.windowY = windowY;
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  @Nonnull
  public PoolingLayer setWindowXY(int x, int y) {
    setWindowY(y);
    setWindowX(x);
    return this;
  }

  @Nonnull
  public PoolingLayer setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
    return this;
  }

  @Nonnull
  public PoolingLayer setPaddingXY(int x, int y) {
    setPaddingX(x);
    setPaddingY(y);
    return this;
  }

  public double getAlpha() {
    return alpha;
  }

  public PoolingLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public enum PoolingMode {
    Avg(cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
    Max(cudnnPoolingMode.CUDNN_POOLING_MAX);
    final int id;

    PoolingMode(final int id) {
      this.id = id;
    }
  }
}
