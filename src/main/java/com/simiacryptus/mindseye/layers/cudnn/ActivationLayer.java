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
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnActivationMode;
import jcuda.jcudnn.cudnnNanPropagation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ActivationLayer extends LayerBase
    implements MultiPrecision<ActivationLayer> {
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
      return new SigmoidActivationLayer().setBalanced(false);
    } else if (mode == Mode.RELU.id) {
      return new ReLuActivationLayer();
    } else {
      throw new RuntimeException("Not Implemented");
    }
  }

  @Nullable
  @Override
  public String getName() {
    return String.format("Activation (%s)", mode);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static ActivationLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ActivationLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    @Nonnull
    final int[] inputSize = inputData.getDimensions();
    @Nonnull
    final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(gpu -> {
        @Nullable
        final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()
            && (!inputResult.isAlive() || mode == Mode.RELU.id)) {
          outputTensor = inputTensor;
        } else {
          @Nonnull
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
              inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull
          final CudaMemory outputData = gpu.allocate((long) precision.size * inputDims * length,
              MemoryType.Managed.ifEnabled(), true);
          outputTensor = new CudaTensor(outputData, outputDescriptor, precision);
        }

        @Nonnull
        final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
            cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
        try {
          CudaMemory memory = inputTensor.getMemory(gpu);
          {
            CudaMemory tensorMemory = outputTensor.getMemory(gpu);
            {
              CudaSystem.handle(gpu.cudnnActivationForward(activationDesc.getPtr(), precision.getPointer(getAlpha()),
                  inputTensor.descriptor.getPtr(), memory.getPtr(), precision.getPointer(0.0),
                  outputTensor.descriptor.getPtr(), tensorMemory.getPtr()));
              assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
              memory.dirty();
              tensorMemory.dirty();
              return outputTensor;
            }
          }
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
                CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
                @Nullable
                CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
                assert length == delta.length();
                CudaTensor localOut = outPtr.getDense(gpu);
                CudaTensor passbackTensor = new CudaTensor(
                    gpu.allocate((long) Tensor.length(inputSize) * length * precision.size,
                        MemoryType.Managed.ifEnabled(), false),
                    gpu.newTensorDescriptor(precision, length, inputSize[2], inputSize[1], inputSize[0],
                        inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1),
                    precision);

                @Nonnull
                final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
                    cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
                try {
                  CudaMemory localOutMemory = localOut.getMemory(gpu);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                  CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
                  CudaMemory passbackTensorMemory = passbackTensor.getMemory(gpu);
                  CudaSystem.handle(gpu.cudnnActivationBackward(activationDesc.getPtr(),
                      precision.getPointer(getAlpha()), localOut.descriptor.getPtr(), localOutMemory.getPtr(),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(), inputTensor.descriptor.getPtr(),
                      inputTensorMemory.getPtr(), precision.getPointer(0.0), passbackTensor.descriptor.getPtr(),
                      passbackTensorMemory.getPtr()));
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  com.simiacryptus.ref.wrappers.RefStream
                      .of(localOutMemory, deltaTensorMemory, inputTensorMemory, passbackTensorMemory)
                      .forEach(CudaMemory::dirty);
                  return new CudaTensorList(passbackTensor, length, inputSize, precision);
                } catch (@Nonnull final Throwable e) {
                  throw new ComponentException(
                      "Error apply " + com.simiacryptus.ref.wrappers.RefArrays.toString(inputSize), e);
                }
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
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", getAlpha());
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public enum Mode {
    RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU), SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID);

    public final int id;

    Mode(final int id) {
      this.id = id;
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ActivationLayer addRef() {
    return (ActivationLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ActivationLayer[] addRefs(ActivationLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ActivationLayer::addRef)
        .toArray((x) -> new ActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") ActivationLayer[][] addRefs(ActivationLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ActivationLayer::addRefs)
        .toArray((x) -> new ActivationLayer[x][]);
  }

}
