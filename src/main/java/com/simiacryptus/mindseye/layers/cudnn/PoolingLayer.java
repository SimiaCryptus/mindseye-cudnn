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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnPoolingMode;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

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

  @Nonnull
  public PoolingLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this.addRef();
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

  @Nonnull
  public PoolingLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this.addRef();
  }

  @Nullable
  @Override
  public String getName() {
    return RefString.format("%sPooling [%d/%d x %d/%d]", mode.name(), windowX, windowY, strideX, strideY);
  }

  public int getPaddingX() {
    return paddingX;
  }

  @Nonnull
  public PoolingLayer setPaddingX(final int paddingX) {
    this.paddingX = paddingX;
    return this.addRef();
  }

  public int getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public PoolingLayer setPaddingY(final int paddingY) {
    this.paddingY = paddingY;
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public PoolingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public int getStrideX() {
    return strideX;
  }

  @Nonnull
  public PoolingLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this.addRef();
  }

  public int getStrideY() {
    return strideY;
  }

  @Nonnull
  public PoolingLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this.addRef();
  }

  public int getWindowX() {
    return windowX;
  }

  @Nonnull
  public PoolingLayer setWindowX(final int windowX) {
    this.windowX = windowX;
    return this.addRef();
  }

  public int getWindowY() {
    return windowY;
  }

  @Nonnull
  public PoolingLayer setWindowY(final int windowY) {
    this.windowY = windowY;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static PoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PoolingLayer(json);
  }

  @Nonnull
  public static PoolingLayer getPoolingLayer(int radius, PoolingLayer.PoolingMode mode, String qualifier) {
    String name = RefString.format("%s.pool:%s;%s", qualifier, radius, mode);
    PoolingLayer temp_37_0009 = new PoolingLayer(UUID.nameUUIDFromBytes(name.getBytes()), name);
    PoolingLayer temp_37_0011 = temp_37_0009.setMode(mode);
    PoolingLayer temp_37_0012 = temp_37_0011.setStrideXY(radius, radius);
    PoolingLayer temp_37_0008 = temp_37_0012.setWindowXY(radius, radius);
    temp_37_0012.freeRef();
    temp_37_0011.freeRef();
    temp_37_0009.freeRef();
    return temp_37_0008;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  PoolingLayer[] addRefs(@Nullable PoolingLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PoolingLayer::addRef).toArray((x) -> new PoolingLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  PoolingLayer[][] addRefs(@Nullable PoolingLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PoolingLayer::addRefs)
        .toArray((x) -> new PoolingLayer[x][]);
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
      Layer temp_37_0013 = getCompatibilityLayer();
      Result temp_37_0007 = temp_37_0013.eval(Result.addRefs(inObj));
      temp_37_0013.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_37_0007;
    }
    TensorList temp_37_0014 = inObj[0].getData();
    @Nonnull final int[] rawInputDims = temp_37_0014.getDimensions();

    temp_37_0014.freeRef();
    int correctionX = correct(rawInputDims[0], strideX, windowX);
    int correctionY = correct(rawInputDims[1], strideY, windowY);
    int paddingX = Math.max(0, PoolingLayer.this.paddingX - ((correctionX + 1) / 2));
    int paddingY = Math.max(0, PoolingLayer.this.paddingY - ((correctionY + 1) / 2));
    assert paddingX >= 0;
    assert paddingY >= 0;
    assert correctionX >= 0;
    assert correctionY >= 0;
    @Nullable
    Result input;
    if (correctionX > 0 || correctionY > 0) {
      ImgPaddingLayer temp_37_0010 = new ImgPaddingLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY);
      ImgPaddingLayer temp_37_0015 = temp_37_0010.setPrecision(precision);
      ImgPaddingLayer temp_37_0016 = temp_37_0015.setHorizontalAlign(Alignment.Center);
      ImgPaddingLayer temp_37_0017 = temp_37_0016.setVerticalAlign(Alignment.Center);
      @Nonnull
      Layer paddingLayer = temp_37_0017.setRoundUp(false);
      temp_37_0017.freeRef();
      temp_37_0016.freeRef();
      temp_37_0015.freeRef();
      temp_37_0010.freeRef();
      input = paddingLayer.eval(inObj[0].addRef());
      paddingLayer.freeRef();
    } else {
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
    final CudaTensor outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
          try {
            gpu.initThread();
            @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(mode.id, poolDims,
                windowSize, padding, stride);
            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                MemoryType.Device, false);
            CudaSystem.handle(CudaSystem.cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(),
                inputTensor.descriptor.getPtr(), 4, outputSize));
            assert inputDims[2] == outputSize[1];
            @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, outputSize[0],
                outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3],
                outputSize[2] * outputSize[3], outputSize[3], 1);
            @Nonnull final CudaMemory outputTensor = gpu.allocate((long) precision.size * Tensor.length(outputSize),
                MemoryType.Managed.ifEnabled(), true);
            CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
            assert inputDataMemory != null;
            CudaSystem.handle(
                gpu.cudnnPoolingForward(poolingDesc.getPtr(), precision.getPointer(alpha), inputTensor.descriptor.getPtr(),
                    inputDataMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputTensor.getPtr()));
            inputTensor.freeRef();
            poolingDesc.freeRef();
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            RefUtil.freeRef(inputDataMemory.dirty());
            inputDataMemory.freeRef();
            RefUtil.freeRef(outputTensor.dirty());
            CudaTensor temp_37_0003 = new CudaTensor(outputTensor,
                outputDescriptor, precision);
            return temp_37_0003;
          } catch (@Nonnull final Throwable e) {
            throw new ComponentException("Error processing " + RefArrays.stream(Result.addRefs(inObj)).map(x -> {
              TensorList temp_37_0018 = x.getData();
              String temp_37_0004 = RefArrays.toString(temp_37_0018.getDimensions());
              temp_37_0018.freeRef();
              x.freeRef();
              return temp_37_0004;
            }).reduce((a, b) -> a + ";" + b) + " with " + this.toString(), e);
          }
        }, inputData.addRef(), Result.addRefs(inObj)),
        inputData.addRef());
    ReferenceCounting.freeRefs(inObj);
    try {
      try {
        try {
          return new Result(new CudaTensorList(outputData == null ? null : outputData.addRef(), inputLength,
              new int[]{outputSize[3], outputSize[2], outputSize[1]}, precision), new Result.Accumulator() {
            {
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
              assert error.length() == inputLength;
              if (input.isAlive()) {
                TensorList data = CudaSystem
                    .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                          @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                              inputLength, inputDims[2], inputDims[1], inputDims[0],
                              inputDims[2] * inputDims[1] * inputDims[0], inputDims[1] * inputDims[0], inputDims[0], 1);
                          @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(mode.id,
                              poolDims, windowSize, padding, stride);
                          @Nullable final CudaTensor inputTensor;
                          synchronized (gpu) {
                            inputTensor = gpu.getTensor(inputData.addRef(), precision,
                                MemoryType.Device, true);
                          }
                          @Nullable final CudaTensor errorPtr;
                          synchronized (gpu) {
                            errorPtr = gpu.getTensor(error.addRef(), precision,
                                MemoryType.Device, true);
                          }
                          @Nonnull final CudaMemory passbackBuffer = gpu.allocate(
                              (long) Tensor.length(inputDims) * precision.size * inputLength,
                              MemoryType.Managed.ifEnabled(), true);
                          assert outputData != null;
                          CudaMemory outputDataMemory = outputData.getMemory(gpu);
                          CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
                          CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
                          assert inputDataMemory != null;
                          assert errorPtrMemory != null;
                          assert outputDataMemory != null;
                          CudaSystem.handle(gpu.cudnnPoolingBackward(poolingDesc.getPtr(), precision.getPointer(alpha),
                              outputData.descriptor.getPtr(), outputDataMemory.getPtr(), errorPtr.descriptor.getPtr(),
                              errorPtrMemory.getPtr(), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                              precision.getPointer(0.0), passbackDescriptor.getPtr(), passbackBuffer.getPtr()));
                          errorPtr.freeRef();
                          inputTensor.freeRef();
                          poolingDesc.freeRef();
                          RefUtil.freeRef(outputDataMemory.dirty());
                          outputDataMemory.freeRef();
                          RefUtil.freeRef(errorPtrMemory.dirty());
                          errorPtrMemory.freeRef();
                          RefUtil.freeRef(inputDataMemory.dirty());
                          inputDataMemory.freeRef();
                          RefUtil.freeRef(passbackBuffer.dirty());
                          CudaTensorList temp_37_0006 = new CudaTensorList(
                              new CudaTensor(passbackBuffer,
                                  passbackDescriptor, precision),
                              inputLength, inputDims, precision);
                          return temp_37_0006;
                        }, error.addRef(), outputData == null ? null : outputData.addRef(),
                        inputData.addRef()), error.addRef());
                input.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
                if (null != data)
                  data.freeRef();
              }
              error.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public @SuppressWarnings("unused")
            void _free() {
            }
          }) {

            {
            }

            @Override
            public boolean isAlive() {
              return input.isAlive() || !isFrozen();
            }

            public void _free() {
            }
          };
        } finally {
          if (null != outputData)
            outputData.freeRef();
        }
      } finally {
        inputData.freeRef();
      }
    } finally {
      input.freeRef();
    }
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

  @Nonnull
  public PoolingLayer setWindowXY(int x, int y) {
    RefUtil.freeRef(setWindowY(y));
    RefUtil.freeRef(setWindowX(x));
    return this.addRef();
  }

  @Nonnull
  public PoolingLayer setStrideXY(int x, int y) {
    RefUtil.freeRef(setStrideX(x));
    RefUtil.freeRef(setStrideY(y));
    return this.addRef();
  }

  @Nonnull
  public PoolingLayer setPaddingXY(int x, int y) {
    RefUtil.freeRef(setPaddingX(x));
    RefUtil.freeRef(setPaddingY(y));
    return this.addRef();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PoolingLayer addRef() {
    return (PoolingLayer) super.addRef();
  }

  public enum PoolingMode {
    Avg(cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING), Max(cudnnPoolingMode.CUDNN_POOLING_MAX);

    final int id;

    PoolingMode(final int id) {
      this.id = id;
    }
  }
}
