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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnActivationMode;
import jcuda.jcudnn.cudnnNanPropagation;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Activation layer.
 */
@SuppressWarnings("serial")
public class ActivationLayer extends LayerBase implements MultiPrecision {
  @SuppressWarnings("unused")
  private static final Logger logger = LoggerFactory.getLogger(ActivationLayer.class);
  /**
   * The Mode.
   */
  final int mode;
  private double alpha = 1.0;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  /**
   * Instantiates a new Activation layer.
   *
   * @param id the id
   */
  public ActivationLayer(final int id) {
    mode = id;
  }

  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected ActivationLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    setAlpha(json.getAsJsonPrimitive("alpha").getAsDouble());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  /**
   * Instantiates a new Activation layer.
   *
   * @param mode the mode
   */
  public ActivationLayer(@Nonnull final Mode mode) {
    this(mode.id);
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   */
  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == Mode.SIGMOID.id) {
      SigmoidActivationLayer sigmoidActivationLayer = new SigmoidActivationLayer();
      sigmoidActivationLayer.setBalanced(false);
      return sigmoidActivationLayer;
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

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * From json activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ActivationLayer(json);
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
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result inputResult = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      Result.Accumulator inputAccumulator = inputResult.getAccumulator();
      boolean inputResultAlive = inputResult.isAlive();
      final CudaTensor outPtr = fwd(inputResult, inputData.addRef(), inputSize, length, inputDims);
      CudaTensorList data = new CudaTensorList(outPtr == null ? null : outPtr.addRef(), length, outputSize, precision);
      Result.Accumulator accumulator = new Accumulator(inputData, outPtr, length, inputSize,
          ActivationLayer.this.precision, ActivationLayer.this.getAlpha(), ActivationLayer.this.mode,
          inputAccumulator, inputResultAlive);
      return new Result(data, accumulator, inputResultAlive || !isFrozen());
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply png res " + RefArrays.toString(inputSize), e);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ActivationLayer addRef() {
    return (ActivationLayer) super.addRef();
  }

  @NotNull
  private CudaTensor fwd(Result inputResult, TensorList inputData, int[] inputSize, int length, int inputDims) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, false);
      CudaTensor outputTensor = null;
      if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()
          && (!inputResult.isAlive() || mode == Mode.RELU.id)) {
        RefUtil.freeRef(outputTensor);
        outputTensor = inputTensor.addRef();
      } else {
        final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
            inputSize[1] * inputSize[0], inputSize[0], 1);
        @Nonnull final CudaMemory outputData = gpu.allocate((long) precision.size * inputDims * length,
            MemoryType.Managed.ifEnabled(), true);
        RefUtil.freeRef(outputTensor);
        outputTensor = new CudaTensor(outputData, outputDescriptor, precision);
      }

      @Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
          cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
      try {
        CudaMemory memory = inputTensor.getMemory(gpu.addRef());
        CudaMemory tensorMemory = outputTensor.getMemory(gpu.addRef());
        assert tensorMemory != null;
        assert memory != null;
        CudaSystem.handle(gpu.cudnnActivationForward(activationDesc.getPtr(), precision.getPointer(getAlpha()),
            inputTensor.descriptor.getPtr(), memory.getPtr(), precision.getPointer(0.0),
            outputTensor.descriptor.getPtr(), tensorMemory.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        gpu.freeRef();
        memory.dirty();
        tensorMemory.dirty();
        inputTensor.freeRef();
        activationDesc.freeRef();
        memory.freeRef();
        tensorMemory.freeRef();
        return outputTensor;
      } catch (@Nonnull final Throwable e) {
        RefUtil.freeRef(outputTensor);
        throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
      }
    }, inputData.addRef(), inputResult), inputData);
  }

  /**
   * The enum Mode.
   */
  public enum Mode {
    /**
     * Relu mode.
     */
    RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU),
    /**
     * Sigmoid mode.
     */
    SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID);

    /**
     * The Id.
     */
    public final int id;

    Mode(final int id) {
      this.id = id;
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final CudaTensor outPtr;
    private final int length;
    private final int[] inputSize;
    private Precision precision;
    private double alpha;
    private int mode;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputData   the input data
     * @param outPtr      the out ptr
     * @param length      the length
     * @param inputSize   the input size
     * @param precision   the precision
     * @param alpha       the alpha
     * @param mode        the mode
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(TensorList inputData, CudaTensor outPtr, int length, int[] inputSize, Precision precision,
                       double alpha, int mode, Result.Accumulator accumulator, boolean alive) {
      this.inputData = inputData;
      this.outPtr = outPtr;
      this.length = length;
      this.inputSize = inputSize;
      this.precision = precision;
      this.alpha = alpha;
      this.mode = mode;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (alive) {
        this.accumulator.accept(buffer, CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          @Nullable
          CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
          @Nullable
          CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision, MemoryType.Device,
              true);
          assert delta != null;
          assert length == delta.length();
          assert outPtr != null;
          CudaTensor localOut = outPtr.getDense(gpu.addRef());
          CudaTensor passbackTensor = new CudaTensor(
              gpu.allocate((long) Tensor.length(inputSize) * length * precision.size, MemoryType.Managed.ifEnabled(),
                  false),
              gpu.newTensorDescriptor(precision, length, inputSize[2], inputSize[1], inputSize[0],
                  inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1),
              precision);

          @Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
              cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
          try {
            CudaMemory localOutMemory = localOut.getMemory(gpu.addRef());
            CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
            CudaMemory inputTensorMemory = inputTensor.getMemory(gpu.addRef());
            CudaMemory passbackTensorMemory = passbackTensor.getMemory(gpu.addRef());
            assert passbackTensorMemory != null;
            assert inputTensorMemory != null;
            assert deltaTensorMemory != null;
            assert localOutMemory != null;
            CudaSystem.handle(gpu.cudnnActivationBackward(activationDesc.getPtr(), precision.getPointer(alpha),
                localOut.descriptor.getPtr(), localOutMemory.getPtr(), deltaTensor.descriptor.getPtr(),
                deltaTensorMemory.getPtr(), inputTensor.descriptor.getPtr(), inputTensorMemory.getPtr(),
                precision.getPointer(0.0), passbackTensor.descriptor.getPtr(), passbackTensorMemory.getPtr()));
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            gpu.freeRef();
            inputTensor.freeRef();
            deltaTensor.freeRef();
            localOut.freeRef();
            activationDesc.freeRef();
            RefStream.of(localOutMemory, deltaTensorMemory, inputTensorMemory, passbackTensorMemory)
                .forEach(cudaMemory -> {
                  cudaMemory.dirty();
                  cudaMemory.freeRef();
                });
            return new CudaTensorList(passbackTensor, length, inputSize, precision);
          } catch (@Nonnull final Throwable e) {
            throw new ComponentException("Error apply " + RefArrays.toString(inputSize), e);
          }
        }, inputData.addRef(), delta == null ? null : delta.addRef(), outPtr == null ? null : outPtr.addRef()), delta));
      } else {
        if (null != delta)
          delta.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
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
