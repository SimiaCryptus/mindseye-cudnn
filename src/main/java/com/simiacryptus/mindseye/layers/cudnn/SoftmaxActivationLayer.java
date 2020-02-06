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
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnSoftmaxAlgorithm;
import jcuda.jcudnn.cudnnSoftmaxMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

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
      Layer temp_23_0006 = getCompatibilityLayer();
      Result temp_23_0005 = temp_23_0006.eval(RefUtil.addRefs(inObj));
      temp_23_0006.freeRef();
      RefUtil.freeRef(inObj);
      return temp_23_0005;
    }
    final Result inputResult = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
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
          CudaMemory inputMemory = inputTensor.getMemory(gpu);
          CudaMemory outputMemory = outputTensor.getMemory(gpu);
          assert outputMemory != null;
          assert inputMemory != null;
          CudaSystem.handle(gpu.cudnnSoftmaxForward(algorithm.code, mode.code, precision.getPointer(1.0),
              inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(0.0),
              outputTensor.descriptor.getPtr(), outputMemory.getPtr()));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
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
      }, inputData.addRef()), inputData.addRef());
      try {
        Result.Accumulator accumulator = new Result.Accumulator() {
          {
            inputData.addRef();
            outPtr.addRef();
            inputResult.addRef();
          }

          @Override
          public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
            if (inputResult.isAlive()) {
              final TensorList data = CudaSystem
                  .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                        @Nullable
                        CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                            MemoryType.Device, true);
                        @Nullable
                        CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                            MemoryType.Device, true);
                        assert outPtr != null;
                        CudaTensor localOut = outPtr.getDense(gpu);
                        assert delta != null;
                        CudaTensor passbackTensor = new CudaTensor(
                            gpu.allocate((long) Tensor.length(inputSize) * length * precision.size,
                                MemoryType.Managed.ifEnabled(), false),
                            gpu.newTensorDescriptor(precision, delta.length(), inputSize[2], inputSize[1],
                                inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
                                inputSize[1] * inputSize[0], inputSize[0], 1),
                            precision);

                        try {
                          CudaMemory localOutMemory = localOut.getMemory(gpu);
                          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                          CudaMemory inputMemory = inputTensor.getMemory(gpu);
                          if (null != inputMemory)
                            inputMemory.freeRef();
                          CudaMemory passbackMemory = passbackTensor.getMemory(gpu);

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
                        }
                        localOut.freeRef();
                        deltaTensor.freeRef();
                        inputTensor.freeRef();
                        CudaTensorList temp_23_0004 = new CudaTensorList(
                            passbackTensor.addRef(), length, inputSize, precision);
                        passbackTensor.freeRef();
                        return temp_23_0004;
                      }, inputData.addRef(), delta == null ? null : delta.addRef(),
                      outPtr == null ? null : outPtr.addRef()), delta == null ? null : delta.addRef());
              inputResult.accumulate(buffer == null ? null : buffer.addRef(),
                  data == null ? null : data.addRef());
              if (null != data)
                data.freeRef();
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
            inputResult.freeRef();
          }
        };
        return new Result(
            new CudaTensorList(outPtr == null ? null : outPtr.addRef(), length, outputSize, precision),
            accumulator) {

          {
            inputResult.addRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
            super._free();
            inputResult.freeRef();
          }

          @Override
          public boolean isAlive() {
            return inputResult.isAlive() || !isFrozen();
          }

          @Override
          public final void accumulate(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
            Result.Accumulator temp_23_0007 = getAccumulator();
            assert temp_23_0007 != null;
            temp_23_0007.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
            temp_23_0007.freeRef();
            if (null != delta)
              delta.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }
        };
      } finally {
        if (null != outPtr)
          outPtr.freeRef();
        inputData.freeRef();
        inputResult.freeRef();
      }
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

}
