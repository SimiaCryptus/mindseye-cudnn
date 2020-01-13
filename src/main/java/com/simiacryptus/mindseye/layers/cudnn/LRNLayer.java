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
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnLRNMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class LRNLayer extends LayerBase implements MultiPrecision<LRNLayer> {
  private static final Logger log = LoggerFactory.getLogger(LRNLayer.class);

  private int width;
  private double alpha;
  private double beta;
  private double k;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private LRNLayer() {
  }

  public LRNLayer(int width) {
    this(width, 1e-4, 0.75, 2.0);
  }

  public LRNLayer(int width, double alpha, double beta, double k) {
    this.setWidth(width);
    RefUtil.freeRef(this.setAlpha(alpha));
    RefUtil.freeRef(this.setBeta(beta));
    RefUtil.freeRef(this.setK(k));
  }

  protected LRNLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    setWidth(json.get("width").getAsInt());
    RefUtil.freeRef(setAlpha(json.get("alpha").getAsDouble()));
    RefUtil.freeRef(setBeta(json.get("beta").getAsDouble()));
    RefUtil.freeRef(setK(json.get("k").getAsDouble()));
    JsonPrimitive precision = json.getAsJsonPrimitive("precision");
    if (null != precision) {
      RefUtil.freeRef(this.setPrecision(Precision.valueOf(precision.getAsString())));
    } else {
      RefUtil.freeRef(this.setPrecision(CudaSettings.INSTANCE().defaultPrecision));
    }
    assert 0 < getWidth();
    assert 0 < getAlpha();
  }

  public double getAlpha() {
    return alpha;
  }

  public LRNLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this.addRef();
  }

  public double getBeta() {
    return beta;
  }

  public LRNLayer setBeta(double beta) {
    this.beta = beta;
    return this.addRef();
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
  }

  public double getK() {
    return k;
  }

  public LRNLayer setK(double k) {
    this.k = k;
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public LRNLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public int getWidth() {
    return width;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  @SuppressWarnings("unused")
  public static LRNLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LRNLayer(json, rs);
  }

  public static @SuppressWarnings("unused") LRNLayer[] addRefs(LRNLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayer::addRef).toArray((x) -> new LRNLayer[x]);
  }

  public static @SuppressWarnings("unused") LRNLayer[][] addRefs(LRNLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayer::addRefs).toArray((x) -> new LRNLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_47_0008 = getCompatibilityLayer();
      Result temp_47_0007 = temp_47_0008.eval(Result.addRefs(inObj));
      if (null != temp_47_0008)
        temp_47_0008.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_47_0007;
    }
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList inputData = input.getData();
    @Nonnull
    final int[] inputSize = inputData.getDimensions();
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    @Nonnull
    final int[] outputSize = new int[] { length, inputSize[2], inputSize[1], inputSize[0] };
    final LRNLayer lrnLayer = this.addRef();
    final CudaTensor outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
      try {
        gpu.initThread();
        @Nonnull
        final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(lrnLayer.getWidth(),
            lrnLayer.getAlpha(), lrnLayer.getBeta(), lrnLayer.getK());
        @Nullable
        final CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), getPrecision(),
            MemoryType.Device, false);
        @Nonnull
        final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(getPrecision(), outputSize[0],
            outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3],
            outputSize[2] * outputSize[3], outputSize[3], 1);
        @Nonnull
        final CudaMemory outputTensor = gpu.allocate((long) getPrecision().size * Tensor.length(outputSize),
            MemoryType.Managed.ifEnabled(), true);
        CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
        CudaSystem
            .handle(gpu.cudnnLRNCrossChannelForward(descriptor.getPtr(), cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1,
                getPrecision().getPointer(1.0), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                getPrecision().getPointer(0.0), outputDescriptor.getPtr(), outputTensor.getPtr()));
        if (null != inputTensor)
          inputTensor.freeRef();
        descriptor.freeRef();
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        RefUtil.freeRef(inputDataMemory.dirty());
        if (null != inputDataMemory)
          inputDataMemory.freeRef();
        RefUtil.freeRef(outputTensor.dirty());
        CudaTensor temp_47_0004 = new CudaTensor(outputTensor == null ? null : outputTensor,
            outputDescriptor == null ? null : outputDescriptor, getPrecision());
        return temp_47_0004;
      } catch (@Nonnull final Throwable e) {
        throw new ComponentException("Error", e);
      }
    }, inputData == null ? null : inputData.addRef(), lrnLayer == null ? null : lrnLayer.addRef()),
        inputData == null ? null : inputData.addRef());
    try {
      try {
        try {
          try {
            return new Result(
                new CudaTensorList(outputData == null ? null : outputData.addRef(), length,
                    new int[] { outputSize[3], outputSize[2], outputSize[1] }, getPrecision()),
                new Result.Accumulator() {
                  {
                  }

                  @Override
                  public void accept(DeltaSet<UUID> buffer, TensorList error) {
                    assert error.length() == inputData.length();
                    if (input.isAlive()) {
                      TensorList data = CudaSystem
                          .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                            @Nonnull
                            final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                                LRNLayer.this.getPrecision(), length, inputSize[2], inputSize[1], inputSize[0],
                                inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0],
                                1);
                            @Nonnull
                            final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(
                                lrnLayer.getWidth(), lrnLayer.getAlpha(), lrnLayer.getBeta(), lrnLayer.getK());
                            @Nullable
                            final CudaTensor inputTensor;
                            synchronized (gpu) {
                              inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(),
                                  LRNLayer.this.getPrecision(), MemoryType.Device, true);
                            }
                            @Nullable
                            final CudaTensor errorPtr;
                            synchronized (gpu) {
                              errorPtr = gpu.getTensor(error == null ? null : error.addRef(),
                                  LRNLayer.this.getPrecision(), MemoryType.Device, true);
                            }
                            @Nonnull
                            final CudaMemory passbackBuffer = gpu.allocate(
                                (long) inputDims * LRNLayer.this.getPrecision().size * length,
                                MemoryType.Managed.ifEnabled(), true);
                            CudaMemory outputDataMemory = outputData.getMemory(gpu);
                            CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
                            CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
                            CudaSystem.handle(gpu.cudnnLRNCrossChannelBackward(descriptor.getPtr(),
                                cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1, LRNLayer.this.getPrecision().getPointer(1.0),
                                outputData.descriptor.getPtr(), outputDataMemory.getPtr(), errorPtr.descriptor.getPtr(),
                                errorPtrMemory.getPtr(), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                                LRNLayer.this.getPrecision().getPointer(0.0), passbackDescriptor.getPtr(),
                                passbackBuffer.getPtr()));
                            if (null != errorPtr)
                              errorPtr.freeRef();
                            if (null != inputTensor)
                              inputTensor.freeRef();
                            descriptor.freeRef();
                            RefUtil.freeRef(outputDataMemory.dirty());
                            if (null != outputDataMemory)
                              outputDataMemory.freeRef();
                            RefUtil.freeRef(errorPtrMemory.dirty());
                            if (null != errorPtrMemory)
                              errorPtrMemory.freeRef();
                            RefUtil.freeRef(inputDataMemory.dirty());
                            if (null != inputDataMemory)
                              inputDataMemory.freeRef();
                            RefUtil.freeRef(passbackBuffer.dirty());
                            CudaTensorList temp_47_0006 = new CudaTensorList(
                                new CudaTensor(passbackBuffer == null ? null : passbackBuffer,
                                    passbackDescriptor == null ? null : passbackDescriptor,
                                    LRNLayer.this.getPrecision()),
                                length, inputSize, LRNLayer.this.getPrecision());
                            return temp_47_0006;
                          }, outputData == null ? null : outputData.addRef(), error == null ? null : error.addRef(),
                              inputData == null ? null : inputData.addRef(),
                              lrnLayer == null ? null : lrnLayer.addRef()), error == null ? null : error.addRef());
                      input.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
                      if (null != data)
                        data.freeRef();
                    }
                    if (null != error)
                      error.freeRef();
                    if (null != buffer)
                      buffer.freeRef();
                  }

                  public @SuppressWarnings("unused") void _free() {
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
          if (null != lrnLayer)
            lrnLayer.freeRef();
        }
      } finally {
        if (null != inputData)
          inputData.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", getAlpha());
    json.addProperty("beta", getBeta());
    json.addProperty("k", getK());
    json.addProperty("width", getWidth());
    json.addProperty("precision", getPrecision().name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") LRNLayer addRef() {
    return (LRNLayer) super.addRef();
  }
}
