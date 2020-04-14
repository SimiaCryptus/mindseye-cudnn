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
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Square activation layer.
 */
@SuppressWarnings("serial")
public class SquareActivationLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private double alpha = 1.0;

  /**
   * Instantiates a new Square activation layer.
   */
  public SquareActivationLayer() {
  }

  /**
   * Instantiates a new Square activation layer.
   *
   * @param id the id
   */
  protected SquareActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    this.alpha = id.getAsJsonPrimitive("alpha").getAsDouble();
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
    return this.as(ProductInputsLayer.class);
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
   * From json square activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the square activation layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SquareActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SquareActivationLayer(json);
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
    int inLength = inObj.length;
    if (inLength != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("inObj.length=" + inLength);
    }
    Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != dimensions.length) {
      input.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    CudaTensorList data = fwd(inputData, dimensions, length);
    Accumulator accumulator = new Accumulator(alpha, precision, length, dimensions, input.getAccumulator(), input.getData(), input.isAlive());
    input.freeRef();
    boolean alive = alive(inObj);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("alpha", alpha);
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
  SquareActivationLayer addRef() {
    return (SquareActivationLayer) super.addRef();
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  @NotNull
  private CudaTensorList fwd(TensorList inputData, int[] dimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
          dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
          dimensions[1] * dimensions[0], dimensions[0], 1);
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      //assert inputTensor.size == rPtr.size;
      @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
          MemoryType.Device, true);
      CudaMemory lPtrMemory = inputTensor.getMemory(gpu.addRef());
      assert lPtrMemory != null;
      CudaSystem.handle(
          gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(alpha), inputTensor.descriptor.getPtr(),
              lPtrMemory.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
              lPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
      inputTensor.freeRef();
      opDescriptor.freeRef();
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      gpu.freeRef();
      outputPtr.dirty();
      lPtrMemory.dirty();
      lPtrMemory.freeRef();
      outputPtr.dirty();
      return new CudaTensorList(
          new CudaTensor(outputPtr, outputDescriptor, precision),
          length, dimensions, precision);
    }, inputData.addRef()), inputData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final int length;
    private final int[] dimensions;
    private double alpha;
    private Precision precision;
    private Result.Accumulator accumulator;
    private boolean alive;
    private @NotNull TensorList inputData;

    /**
     * Instantiates a new Accumulator.
     *
     * @param alpha       the alpha
     * @param precision   the precision
     * @param length      the length
     * @param dimensions  the dimensions
     * @param accumulator the accumulator
     * @param inputData   the input data
     * @param alive       the alive
     */
    public Accumulator(double alpha, Precision precision, int length, int[] dimensions, Result.Accumulator accumulator, @NotNull TensorList inputData, boolean alive) {
      this.length = length;
      this.dimensions = dimensions;
      this.alpha = alpha;
      this.precision = precision;
      this.accumulator = accumulator;
      this.alive = alive;
      this.inputData = inputData;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (alive) {
        //assert deltaTensor.size == inputTensor.size;
        TensorList delta1 = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                      .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                  final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
                      length, dimensions[2], dimensions[1], dimensions[0],
                      dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                      dimensions[0], 1);
                  @Nullable final CudaTensor deltaTensor = gpu.getTensor(
                      delta == null ? null : delta.addRef(), precision, MemoryType.Device, true);
                  @Nullable final CudaTensor inputTensor = gpu.getTensor(
                      inputData.addRef(), precision, MemoryType.Device, false);
                  //assert deltaTensor.size == inputTensor.size;
                  @Nonnull final CudaMemory outputPtr = gpu.allocate(
                      (long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
                  CudaMemory rightTensorMemory = inputTensor.getMemory(gpu.addRef());
                  assert rightTensorMemory != null;
                  assert deltaTensorMemory != null;
                  CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(2),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(), precision.getPointer(alpha),
                      inputTensor.descriptor.getPtr(), rightTensorMemory.getPtr(), precision.getPointer(0.0),
                      outputDescriptor.getPtr(), outputPtr.getPtr()));
                  gpu.freeRef();
                  inputTensor.freeRef();
                  deltaTensor.freeRef();
                  opDescriptor.freeRef();
                  deltaTensorMemory.dirty();
                  deltaTensorMemory.freeRef();
                  rightTensorMemory.dirty();
                  rightTensorMemory.freeRef();
                  outputPtr.dirty();
                  return new CudaTensorList(
                      new CudaTensor(outputPtr, outputDescriptor, precision),
                      length, dimensions, precision);
                },
                delta == null ? null : delta.addRef(),
                inputData.addRef()
            ), delta);
        this.accumulator.accept(buffer, delta1);
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
      accumulator.freeRef();
      inputData.freeRef();
    }
  }
}
