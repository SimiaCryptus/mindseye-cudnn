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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnLRNMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class LRNLayer extends LayerBase implements MultiPrecision {
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
    setAlpha(alpha);
    setBeta(beta);
    setK(k);
  }

  protected LRNLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    setWidth(json.get("width").getAsInt());
    setAlpha(json.get("alpha").getAsDouble());
    setBeta(json.get("beta").getAsDouble());
    setK(json.get("k").getAsDouble());
    JsonPrimitive precision = json.getAsJsonPrimitive("precision");
    if (null != precision) {
      setPrecision(Precision.valueOf(precision.getAsString()));
    } else {
      setPrecision(CudaSettings.INSTANCE().defaultPrecision);
    }
    assert 0 < getWidth();
    assert 0 < getAlpha();
  }

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public double getBeta() {
    return beta;
  }

  public void setBeta(double beta) {
    this.beta = beta;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
  }

  public double getK() {
    return k;
  }

  public void setK(double k) {
    this.k = k;
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

  public int getWidth() {
    return width;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static LRNLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LRNLayer(json, rs);
  }


  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_47_0008 = getCompatibilityLayer();
      Result temp_47_0007 = temp_47_0008.eval(RefUtil.addRefs(inObj));
      temp_47_0008.freeRef();
      RefUtil.freeRef(inObj);
      return temp_47_0007;
    }
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    @Nonnull final int[] outputSize = new int[]{length, inputSize[2], inputSize[1], inputSize[0]};
    final LRNLayer lrnLayer = this.addRef();
    final CudaTensor outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
          try {
            gpu.initThread();
            @Nonnull final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(lrnLayer.getWidth(),
                lrnLayer.getAlpha(), lrnLayer.getBeta(), lrnLayer.getK());
            assert getPrecision() != null;
            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), getPrecision(),
                MemoryType.Device, false);
            @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(getPrecision(), outputSize[0],
                outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3],
                outputSize[2] * outputSize[3], outputSize[3], 1);
            @Nonnull final CudaMemory outputTensor = gpu.allocate((long) getPrecision().size * Tensor.length(outputSize),
                MemoryType.Managed.ifEnabled(), true);
            CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
            assert inputDataMemory != null;
            CudaSystem
                .handle(gpu.cudnnLRNCrossChannelForward(descriptor.getPtr(), cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    getPrecision().getPointer(1.0), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                    getPrecision().getPointer(0.0), outputDescriptor.getPtr(), outputTensor.getPtr()));
            inputTensor.freeRef();
            descriptor.freeRef();
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            inputDataMemory.dirty();
            inputDataMemory.freeRef();
            outputTensor.dirty();
            CudaTensor temp_47_0004 = new CudaTensor(outputTensor,
                outputDescriptor, getPrecision());
            return temp_47_0004;
          } catch (@Nonnull final Throwable e) {
            throw new ComponentException("Error", e);
          }
        }, inputData.addRef(), lrnLayer.addRef()),
        inputData.addRef());
    try {
      try {
        try {
          try {
            assert getPrecision() != null;
            Result.Accumulator accumulator = new Result.Accumulator() {
              {
                outputData.addRef();
                input.addRef();
                inputData.addRef();
                lrnLayer.addRef();
              }

              @Override
              public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
                assert error.length() == inputData.length();
                if (input.isAlive()) {
                  TensorList data = CudaSystem
                      .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                            @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                                LRNLayer.this.getPrecision(), length, inputSize[2], inputSize[1], inputSize[0],
                                inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0],
                                1);
                            @Nonnull final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(
                                lrnLayer.getWidth(), lrnLayer.getAlpha(), lrnLayer.getBeta(), lrnLayer.getK());
                            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(),
                                LRNLayer.this.getPrecision(), MemoryType.Device, true);
                            @Nullable CudaTensor errorPtr = gpu.getTensor(error.addRef(),
                                LRNLayer.this.getPrecision(), MemoryType.Device, true);
                            @Nonnull final CudaMemory passbackBuffer = gpu.allocate(
                                (long) inputDims * LRNLayer.this.getPrecision().size * length,
                                MemoryType.Managed.ifEnabled(), true);
                            assert outputData != null;
                            CudaMemory outputDataMemory = outputData.getMemory(gpu);
                            CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
                            CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
                            assert inputDataMemory != null;
                            assert errorPtrMemory != null;
                            assert outputDataMemory != null;
                            CudaSystem.handle(gpu.cudnnLRNCrossChannelBackward(descriptor.getPtr(),
                                cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1, LRNLayer.this.getPrecision().getPointer(1.0),
                                outputData.descriptor.getPtr(), outputDataMemory.getPtr(), errorPtr.descriptor.getPtr(),
                                errorPtrMemory.getPtr(), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                                LRNLayer.this.getPrecision().getPointer(0.0), passbackDescriptor.getPtr(),
                                passbackBuffer.getPtr()));
                            errorPtr.freeRef();
                            inputTensor.freeRef();
                            descriptor.freeRef();
                            outputDataMemory.dirty();
                            outputDataMemory.freeRef();
                            errorPtrMemory.dirty();
                            errorPtrMemory.freeRef();
                            inputDataMemory.dirty();
                            inputDataMemory.freeRef();
                            passbackBuffer.dirty();
                            CudaTensorList temp_47_0006 = new CudaTensorList(
                                new CudaTensor(passbackBuffer,
                                    passbackDescriptor,
                                    LRNLayer.this.getPrecision()),
                                length, inputSize, LRNLayer.this.getPrecision());
                            return temp_47_0006;
                          }, outputData == null ? null : outputData.addRef(), error.addRef(),
                          inputData.addRef(),
                          lrnLayer.addRef()), error.addRef());
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
                super._free();
                outputData.freeRef();
                input.freeRef();
                inputData.freeRef();
                lrnLayer.freeRef();
              }
            };
            return new Result(
                new CudaTensorList(outputData == null ? null : outputData.addRef(), length,
                    new int[]{outputSize[3], outputSize[2], outputSize[1]}, getPrecision()),
                accumulator) {
              {
                input.addRef();
              }

              @Override
              public boolean isAlive() {
                return input.isAlive() || !isFrozen();
              }

              public @SuppressWarnings("unused")
              void _free() {
                super._free();
                input.freeRef();
              }
            };
          } finally {
            if (null != outputData)
              outputData.freeRef();
          }
        } finally {
          lrnLayer.freeRef();
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
    json.addProperty("alpha", getAlpha());
    json.addProperty("beta", getBeta());
    json.addProperty("k", getK());
    json.addProperty("width", getWidth());
    assert getPrecision() != null;
    json.addProperty("precision", getPrecision().name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LRNLayer addRef() {
    return (LRNLayer) super.addRef();
  }
}
