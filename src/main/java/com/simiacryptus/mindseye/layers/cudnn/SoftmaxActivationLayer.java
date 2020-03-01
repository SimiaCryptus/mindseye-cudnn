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
import com.simiacryptus.mindseye.layers.java.ImgPixelSoftmaxLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnSoftmaxAlgorithm;
import jcuda.jcudnn.cudnnSoftmaxMode;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  private SoftmaxAlgorithm algorithm = SoftmaxAlgorithm.ACCURATE;
  private SoftmaxMode mode = SoftmaxMode.INSTANCE;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public SoftmaxActivationLayer() {
  }

  protected SoftmaxActivationLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    algorithm = SoftmaxAlgorithm.valueOf(json.get("algorithm").getAsString());
    mode = SoftmaxMode.valueOf(json.get("mode").getAsString());
  }

  public SoftmaxAlgorithm getAlgorithm() {
    return algorithm;
  }

  public void setAlgorithm(SoftmaxAlgorithm algorithm) {
    this.algorithm = algorithm;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    assert algorithm != SoftmaxAlgorithm.LOG;
    if (mode == SoftmaxMode.CHANNEL)
      return this.as(ImgPixelSoftmaxLayer.class);
    return this.as(SoftmaxLayer.class);
  }

  public SoftmaxMode getMode() {
    return mode;
  }

  public void setMode(SoftmaxMode mode) {
    this.mode = mode;
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

  @Nonnull
  @SuppressWarnings("unused")
  public static SoftmaxActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
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
    final Result inputResult = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = fwd(inputData.addRef(), inputSize, length, inputDims);
      boolean alive = inputResult.isAlive();
      Result.Accumulator accumulator = new Accumulator(algorithm, mode, precision, inputData, outPtr.addRef(), inputSize, length, inputResult.getAccumulator(), inputResult.isAlive());
      inputResult.freeRef();
      CudaTensorList data = new CudaTensorList(outPtr, length, outputSize, precision);
      return new Result(data, accumulator, alive);
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply png res " + RefArrays.toString(inputSize), e);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("algorithm", algorithm.name());
    json.addProperty("mode", mode.name());
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
  SoftmaxActivationLayer addRef() {
    return (SoftmaxActivationLayer) super.addRef();
  }

  @NotNull
  private CudaTensor fwd(TensorList inputData, int[] inputSize, int length, int inputDims) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
      @Nullable
      CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      CudaTensor outputTensor = null;
      if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()) {
        RefUtil.freeRef(outputTensor);
        outputTensor = inputTensor.addRef();
      } else {
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
            inputSize[1] * inputSize[0], inputSize[0], 1);
        @Nonnull final CudaMemory outputData = gpu.allocate(precision.size * 1l * inputDims * length,
            MemoryType.Managed.ifEnabled(), true);
        RefUtil.freeRef(outputTensor);
        outputTensor = new CudaTensor(outputData,
            outputDescriptor, precision);
      }
      try {
        CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());
        CudaMemory outputMemory = outputTensor.getMemory(gpu.addRef());
        assert outputMemory != null;
        assert inputMemory != null;
        CudaSystem.handle(gpu.cudnnSoftmaxForward(algorithm.code, mode.code, precision.getPointer(1.0),
            inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(0.0),
            outputTensor.descriptor.getPtr(), outputMemory.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        gpu.freeRef();
        inputMemory.dirty();
        inputMemory.freeRef();
        outputMemory.dirty();
        outputMemory.freeRef();
        inputTensor.freeRef();
        return outputTensor;
      } catch (@Nonnull final Throwable e) {
        throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
      } finally {
      }
    }, inputData.addRef()), inputData);
  }

  public enum SoftmaxAlgorithm {
    FAST(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_FAST), ACCURATE(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE),
    LOG(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_LOG);

    public final int code;

    SoftmaxAlgorithm(final int code) {
      this.code = code;
    }
  }

  public enum SoftmaxMode {
    CHANNEL(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL), INSTANCE(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_INSTANCE);

    public final int code;

    SoftmaxMode(final int code) {
      this.code = code;
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final CudaTensor outPtr;
    private final int[] inputSize;
    private final int length;
    private SoftmaxAlgorithm algorithm;
    private SoftmaxMode mode;
    private Precision precision;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(SoftmaxAlgorithm algorithm, SoftmaxMode mode, Precision precision, TensorList inputData, CudaTensor outPtr, int[] inputSize, int length, Result.Accumulator accumulator, boolean alive) {
      this.inputData = inputData;
      this.outPtr = outPtr;
      this.inputSize = inputSize;
      this.length = length;
      this.algorithm = algorithm;
      this.mode = mode;
      this.precision = precision;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (alive) {
        final TensorList data = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nullable
                  CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                      MemoryType.Device, true);
                  @Nullable
                  CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                      MemoryType.Device, true);
                  assert outPtr != null;
                  CudaTensor localOut = outPtr.getDense(gpu.addRef());
                  assert delta != null;
                  CudaTensor passbackTensor = new CudaTensor(
                      gpu.allocate((long) Tensor.length(inputSize) * length * precision.size,
                          MemoryType.Managed.ifEnabled(), false),
                      gpu.newTensorDescriptor(precision, delta.length(), inputSize[2], inputSize[1],
                          inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
                          inputSize[1] * inputSize[0], inputSize[0], 1),
                      precision);

                  try {
                    CudaMemory localOutMemory = localOut.getMemory(gpu.addRef());
                    CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
                    CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());
                    if (null != inputMemory)
                      inputMemory.freeRef();
                    CudaMemory passbackMemory = passbackTensor.getMemory(gpu.addRef());

                    assert passbackMemory != null;
                    assert deltaTensorMemory != null;
                    assert localOutMemory != null;
                    CudaSystem.handle(gpu.cudnnSoftmaxBackward(algorithm.code, mode.code,
                        precision.getPointer(1.0), localOut.descriptor.getPtr(), localOutMemory.getPtr(),
                        deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                        precision.getPointer(0.0), passbackTensor.descriptor.getPtr(),
                        passbackMemory.getPtr()));
                    localOutMemory.dirty();
                    localOutMemory.freeRef();
                    deltaTensorMemory.dirty();
                    deltaTensorMemory.freeRef();
                    passbackMemory.dirty();
                    passbackMemory.freeRef();
                  } catch (@Nonnull final Throwable e) {
                    throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
                  } finally {
                    gpu.freeRef();
                  }
                  localOut.freeRef();
                  deltaTensor.freeRef();
                  inputTensor.freeRef();
                  return new CudaTensorList(
                      passbackTensor, length, inputSize, precision);
                }, inputData.addRef(), delta == null ? null : delta.addRef(),
                outPtr == null ? null : outPtr.addRef()), delta == null ? null : delta.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, data);
      }
      if (null != delta)
        delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      inputData.freeRef();
      outPtr.freeRef();
      accumulator.freeRef();
    }
  }
}
