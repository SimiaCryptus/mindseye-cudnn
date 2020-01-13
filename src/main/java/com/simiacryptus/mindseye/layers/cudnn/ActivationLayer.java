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
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.ref.wrappers.RefString;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnActivationMode;
import jcuda.jcudnn.cudnnNanPropagation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ActivationLayer extends LayerBase implements MultiPrecision<ActivationLayer> {
  @SuppressWarnings("unused")
  private static final Logger logger = LoggerFactory.getLogger(ActivationLayer.class);
  final int mode;
  private double alpha = 1.0;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public ActivationLayer(final int id) {
    mode = id;
  }

  protected ActivationLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    setAlpha(json.getAsJsonPrimitive("alpha").getAsDouble());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  public ActivationLayer(@Nonnull final Mode mode) {
    this(mode.id);
  }

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == Mode.SIGMOID.id) {
      SigmoidActivationLayer temp_50_0007 = new SigmoidActivationLayer();
      SigmoidActivationLayer temp_50_0006 = temp_50_0007.setBalanced(false);
      if (null != temp_50_0007)
        temp_50_0007.freeRef();
      return temp_50_0006;
    } else if (mode == Mode.RELU.id) {
      return new ReLuActivationLayer();
    } else {
      throw new RuntimeException("Not Implemented");
    }
  }

  @Nullable
  @Override
  public String getName() {
    return RefString.format("Activation (%s)", mode);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ActivationLayer(json);
  }

  public static @SuppressWarnings("unused") ActivationLayer[] addRefs(ActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayer::addRef)
        .toArray((x) -> new ActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") ActivationLayer[][] addRefs(ActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayer::addRefs)
        .toArray((x) -> new ActivationLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_50_0008 = getCompatibilityLayer();
      Result temp_50_0005 = temp_50_0008.eval(Result.addRefs(inObj));
      if (null != temp_50_0008)
        temp_50_0008.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_50_0005;
    }
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
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
        final CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
            MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()
            && (!inputResult.isAlive() || mode == Mode.RELU.id)) {
          outputTensor = inputTensor == null ? null : inputTensor.addRef();
        } else {
          @Nonnull
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
              inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull
          final CudaMemory outputData = gpu.allocate((long) precision.size * inputDims * length,
              MemoryType.Managed.ifEnabled(), true);
          outputTensor = new CudaTensor(outputData == null ? null : outputData,
              outputDescriptor == null ? null : outputDescriptor, precision);
        }

        @Nonnull
        final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
            cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
        try {
          CudaMemory memory = inputTensor.getMemory(gpu);
          CudaMemory tensorMemory = outputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnActivationForward(activationDesc.getPtr(), precision.getPointer(getAlpha()),
              inputTensor.descriptor.getPtr(), memory.getPtr(), precision.getPointer(0.0),
              outputTensor.descriptor.getPtr(), tensorMemory.getPtr()));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          RefUtil.freeRef(memory.dirty());
          RefUtil.freeRef(tensorMemory.dirty());
          if (null != inputTensor)
            inputTensor.freeRef();
          activationDesc.freeRef();
          if (null != memory)
            memory.freeRef();
          if (null != tensorMemory)
            tensorMemory.freeRef();
          return outputTensor;
        } catch (@Nonnull final Throwable e) {
          throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
        } finally {
        }
      }, inputData == null ? null : inputData.addRef(), inputResult == null ? null : inputResult.addRef()),
          inputData == null ? null : inputData.addRef());
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
                            CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(),
                                precision, MemoryType.Device, true);
                            @Nullable
                            CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                                MemoryType.Device, true);
                            assert length == delta.length();
                            CudaTensor localOut = outPtr.getDense(gpu);
                            CudaTensor passbackTensor = new CudaTensor(
                                gpu.allocate((long) Tensor.length(inputSize) * length * precision.size,
                                    MemoryType.Managed.ifEnabled(), false),
                                gpu.newTensorDescriptor(precision, length, inputSize[2], inputSize[1], inputSize[0],
                                    inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0],
                                    inputSize[0], 1),
                                precision);

                            @Nonnull
                            final CudaResource<cudnnActivationDescriptor> activationDesc = gpu
                                .newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
                            try {
                              CudaMemory localOutMemory = localOut.getMemory(gpu);
                              CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                              CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
                              CudaMemory passbackTensorMemory = passbackTensor.getMemory(gpu);
                              CudaSystem.handle(gpu.cudnnActivationBackward(activationDesc.getPtr(),
                                  precision.getPointer(ActivationLayer.this.getAlpha()), localOut.descriptor.getPtr(),
                                  localOutMemory.getPtr(), deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                                  inputTensor.descriptor.getPtr(), inputTensorMemory.getPtr(),
                                  precision.getPointer(0.0), passbackTensor.descriptor.getPtr(),
                                  passbackTensorMemory.getPtr()));
                              assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                              RefStream
                                  .of(localOutMemory == null ? null : localOutMemory.addRef(),
                                      deltaTensorMemory == null ? null : deltaTensorMemory.addRef(),
                                      inputTensorMemory == null ? null : inputTensorMemory.addRef(),
                                      passbackTensorMemory == null ? null : passbackTensorMemory.addRef())
                                  .forEach(CudaMemory::dirty);
                              if (null != passbackTensorMemory)
                                passbackTensorMemory.freeRef();
                              if (null != inputTensorMemory)
                                inputTensorMemory.freeRef();
                              if (null != deltaTensorMemory)
                                deltaTensorMemory.freeRef();
                              if (null != localOutMemory)
                                localOutMemory.freeRef();
                              if (null != inputTensor)
                                inputTensor.freeRef();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              if (null != localOut)
                                localOut.freeRef();
                              CudaTensorList temp_50_0004 = new CudaTensorList(
                                  passbackTensor == null ? null : passbackTensor.addRef(), length, inputSize,
                                  precision);
                              if (null != passbackTensor)
                                passbackTensor.freeRef();
                              activationDesc.freeRef();
                              return temp_50_0004;
                            } catch (@Nonnull final Throwable e) {
                              throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
                            }
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
                Result.Accumulator temp_50_0009 = getAccumulator();
                temp_50_0009.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                if (null != temp_50_0009)
                  temp_50_0009.freeRef();
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
    json.addProperty("alpha", getAlpha());
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ActivationLayer addRef() {
    return (ActivationLayer) super.addRef();
  }

  public enum Mode {
    RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU), SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID);

    public final int id;

    Mode(final int id) {
      this.id = id;
    }
  }

}
