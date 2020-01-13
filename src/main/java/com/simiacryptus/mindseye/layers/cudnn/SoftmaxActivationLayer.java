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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnSoftmaxAlgorithm;
import jcuda.jcudnn.cudnnSoftmaxMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends LayerBase implements MultiPrecision<SoftmaxActivationLayer> {
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
    return this.addRef();
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
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SoftmaxActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static SoftmaxActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }

  public static @SuppressWarnings("unused") SoftmaxActivationLayer[] addRefs(SoftmaxActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxActivationLayer::addRef)
        .toArray((x) -> new SoftmaxActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") SoftmaxActivationLayer[][] addRefs(SoftmaxActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxActivationLayer::addRefs)
        .toArray((x) -> new SoftmaxActivationLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_23_0006 = getCompatibilityLayer();
      Result temp_23_0005 = temp_23_0006.eval(Result.addRefs(inObj));
      if (null != temp_23_0006)
        temp_23_0006.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_23_0005;
    }
    final Result inputResult = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList inputData = inputResult.getData();
    @Nonnull
    final int[] inputSize = inputData.getDimensions();
    @Nonnull
    final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
        @Nullable
        CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
            MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()) {
          outputTensor = inputTensor == null ? null : inputTensor.addRef();
        } else {
          @Nonnull
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
              inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull
          final CudaMemory outputData = gpu.allocate(precision.size * 1l * inputDims * length,
              MemoryType.Managed.ifEnabled(), true);
          outputTensor = new CudaTensor(outputData == null ? null : outputData,
              outputDescriptor == null ? null : outputDescriptor, precision);
        }
        try {
          CudaMemory inputMemory = inputTensor.getMemory(gpu);
          CudaMemory outputMemory = outputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnSoftmaxForward(algorithm.code, mode.code, precision.getPointer(1.0),
              inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(0.0),
              outputTensor.descriptor.getPtr(), outputMemory.getPtr()));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          RefUtil.freeRef(inputMemory.dirty());
          if (null != inputMemory)
            inputMemory.freeRef();
          RefUtil.freeRef(outputMemory.dirty());
          if (null != outputMemory)
            outputMemory.freeRef();
          if (null != inputTensor)
            inputTensor.freeRef();
          return outputTensor;
        } catch (@Nonnull final Throwable e) {
          throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
        } finally {
        }
      }, inputData == null ? null : inputData.addRef()), inputData == null ? null : inputData.addRef());
      try {
        try {
          try {
            return new Result(
                new CudaTensorList(outPtr == null ? null : outPtr.addRef(), length, outputSize, precision),
                new Result.Accumulator() {
                  {
                  }

                  @Override
                  public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                    if (inputResult.isAlive()) {
                      final TensorList data = CudaSystem
                          .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                            @Nullable
                            CudaTensor inputTensor;
                            synchronized (gpu) {
                              inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
                                  MemoryType.Device, true);
                            }
                            @Nullable
                            CudaTensor deltaTensor;
                            synchronized (gpu) {
                              deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                                  MemoryType.Device, true);
                            }
                            CudaTensor localOut = outPtr.getDense(gpu);
                            CudaTensor passbackTensor;
                            passbackTensor = new CudaTensor(
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

                              CudaSystem.handle(gpu.cudnnSoftmaxBackward(algorithm.code, mode.code,
                                  precision.getPointer(1.0), localOut.descriptor.getPtr(), localOutMemory.getPtr(),
                                  deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                                  precision.getPointer(0.0), passbackTensor.descriptor.getPtr(),
                                  passbackMemory.getPtr()));
                              RefUtil.freeRef(localOutMemory.dirty());
                              if (null != localOutMemory)
                                localOutMemory.freeRef();
                              RefUtil.freeRef(deltaTensorMemory.dirty());
                              if (null != deltaTensorMemory)
                                deltaTensorMemory.freeRef();
                              RefUtil.freeRef(passbackMemory.dirty());
                              if (null != passbackMemory)
                                passbackMemory.freeRef();
                            } catch (@Nonnull final Throwable e) {
                              throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
                            } finally {
                            }
                            if (null != localOut)
                              localOut.freeRef();
                            if (null != deltaTensor)
                              deltaTensor.freeRef();
                            if (null != inputTensor)
                              inputTensor.freeRef();
                            CudaTensorList temp_23_0004 = new CudaTensorList(
                                passbackTensor == null ? null : passbackTensor.addRef(), length, inputSize, precision);
                            if (null != passbackTensor)
                              passbackTensor.freeRef();
                            return temp_23_0004;
                          }, inputData == null ? null : inputData.addRef(), delta == null ? null : delta.addRef(),
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

                  public @SuppressWarnings("unused") void _free() {
                  }
                }) {

              {
              }

              @Override
              public boolean isAlive() {
                return inputResult.isAlive() || !isFrozen();
              }

              @Override
              public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
                Result.Accumulator temp_23_0007 = getAccumulator();
                temp_23_0007.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                if (null != temp_23_0007)
                  temp_23_0007.freeRef();
                if (null != delta)
                  delta.freeRef();
                if (null != buffer)
                  buffer.freeRef();
              }

              public void _free() {
              }
            };
          } finally {
            if (null != outPtr)
              outPtr.freeRef();
          }
        } finally {
          if (null != inputData)
            inputData.freeRef();
        }
      } finally {
        if (null != inputResult)
          inputResult.freeRef();
      }
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply png res " + RefArrays.toString(inputSize), e);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") SoftmaxActivationLayer addRef() {
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
