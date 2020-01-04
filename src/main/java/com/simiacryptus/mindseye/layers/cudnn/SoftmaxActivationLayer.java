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
import jcuda.jcudnn.cudnnSoftmaxAlgorithm;
import jcuda.jcudnn.cudnnSoftmaxMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class SoftmaxActivationLayer extends LayerBase
    implements MultiPrecision<SoftmaxActivationLayer> {
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

  public SoftmaxActivationLayer setAlgorithm(SoftmaxAlgorithm algorithm) {
    this.algorithm = algorithm;
    return this;
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

  public SoftmaxActivationLayer setMode(SoftmaxMode mode) {
    this.mode = mode;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SoftmaxActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static SoftmaxActivationLayer fromJson(@Nonnull final JsonObject json,
                                                com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }

  public static @SuppressWarnings("unused")
  SoftmaxActivationLayer[] addRefs(SoftmaxActivationLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxActivationLayer::addRef)
        .toArray((x) -> new SoftmaxActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SoftmaxActivationLayer[][] addRefs(SoftmaxActivationLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxActivationLayer::addRefs)
        .toArray((x) -> new SoftmaxActivationLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(gpu -> {
        @Nullable
        CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()) {
          outputTensor = inputTensor;
        } else {
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
              inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull final CudaMemory outputData = gpu.allocate(precision.size * 1l * inputDims * length,
              MemoryType.Managed.ifEnabled(), true);
          outputTensor = new CudaTensor(outputData, outputDescriptor, precision);
        }
        try {
          CudaMemory inputMemory = inputTensor.getMemory(gpu);
          CudaMemory outputMemory = outputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnSoftmaxForward(algorithm.code, mode.code, precision.getPointer(1.0),
              inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(0.0),
              outputTensor.descriptor.getPtr(), outputMemory.getPtr()));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          inputMemory.dirty();
          outputMemory.dirty();
          return outputTensor;
        } catch (@Nonnull final Throwable e) {
          throw new ComponentException("Error apply " + com.simiacryptus.ref.wrappers.RefArrays.toString(inputSize), e);
        } finally {
        }
      }, inputData);
      return new Result(new CudaTensorList(outPtr, length, outputSize, precision),
          (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
            if (inputResult.isAlive()) {
              final TensorList data = CudaSystem.run(gpu -> {
                @Nullable
                CudaTensor inputTensor;
                synchronized (gpu) {
                  inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
                }
                @Nullable
                CudaTensor deltaTensor;
                synchronized (gpu) {
                  deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
                }
                CudaTensor localOut = outPtr.getDense(gpu);
                CudaTensor passbackTensor;
                passbackTensor = new CudaTensor(
                    gpu.allocate((long) Tensor.length(inputSize) * length * precision.size,
                        MemoryType.Managed.ifEnabled(), false),
                    gpu.newTensorDescriptor(precision, delta.length(), inputSize[2], inputSize[1], inputSize[0],
                        inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1),
                    precision);

                try {
                  CudaMemory localOutMemory = localOut.getMemory(gpu);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                  CudaMemory inputMemory = inputTensor.getMemory(gpu);
                  CudaMemory passbackMemory = passbackTensor.getMemory(gpu);

                  CudaSystem.handle(gpu.cudnnSoftmaxBackward(algorithm.code, mode.code, precision.getPointer(1.0),
                      localOut.descriptor.getPtr(), localOutMemory.getPtr(), deltaTensor.descriptor.getPtr(),
                      deltaTensorMemory.getPtr(), precision.getPointer(0.0), passbackTensor.descriptor.getPtr(),
                      passbackMemory.getPtr()));
                  localOutMemory.dirty();
                  deltaTensorMemory.dirty();
                  passbackMemory.dirty();
                } catch (@Nonnull final Throwable e) {
                  throw new ComponentException(
                      "Error apply " + com.simiacryptus.ref.wrappers.RefArrays.toString(inputSize), e);
                } finally {
                }
                return new CudaTensorList(passbackTensor, length, inputSize, precision);
              }, delta);
              inputResult.accumulate(buffer, data);
            }
          }) {

        @Override
        public boolean isAlive() {
          return inputResult.isAlive() || !isFrozen();
        }

        @Override
        public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
          getAccumulator().accept(buffer, delta);
        }

        public void _free() {
        }
      };
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply png res " + com.simiacryptus.ref.wrappers.RefArrays.toString(inputSize),
          e);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("algorithm", algorithm.name());
    json.addProperty("mode", mode.name());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

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
