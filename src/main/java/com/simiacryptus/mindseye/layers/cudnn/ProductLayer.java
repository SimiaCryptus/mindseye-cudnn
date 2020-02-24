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
import com.simiacryptus.ref.wrappers.RefString;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ProductLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean bypassOnError = false;

  public ProductLayer() {
    this(UUID.randomUUID());
  }

  public ProductLayer(UUID id) {
    super(id, "ProductLayer");
  }

  protected ProductLayer(@Nonnull final JsonObject json) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    this.setBypassOnError(json.getAsJsonPrimitive("bypassOnError").getAsBoolean());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(ProductInputsLayer.class);
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

  public boolean isBypassOnError() {
    return bypassOnError;
  }

  public void setBypassOnError(boolean bypassOnError) {
    this.bypassOnError = bypassOnError;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ProductLayer(json);
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
    if (inLength != 2) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("inObj.length=" + inLength);
    }
    Result left = inObj[0].addRef();
    Result right = inObj[1].addRef();
    final TensorList leftData = left.getData();
    final TensorList rightData = right.getData();
    @Nonnull final int[] leftDimensions = leftData.getDimensions();
    @Nonnull final int[] rightDimensions = rightData.getDimensions();
    final int length = leftData.length();
    if (3 != leftDimensions.length) {
      left.freeRef();
      right.freeRef();
      leftData.freeRef();
      rightData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    if (leftDimensions[0] != rightDimensions[0] && leftDimensions[0] != 1 && 1 != rightDimensions[0] || rightDimensions.length > 1 && leftDimensions[1] != rightDimensions[1] && leftDimensions[1] != 1 && 1 != rightDimensions[1] || rightDimensions.length > 2 && leftDimensions[2] != rightDimensions[2] && leftDimensions[2] != 1 && 1 != rightDimensions[2]) {
      if (isBypassOnError()) {
        RefUtil.freeRef(inObj[1].getData());
        left.freeRef();
        right.freeRef();
        leftData.freeRef();
        rightData.freeRef();
        Result temp_26_0011 = inObj[0].addRef();
        RefUtil.freeRef(inObj);
        return temp_26_0011;
      } else {
        left.freeRef();
        right.freeRef();
        leftData.freeRef();
        rightData.freeRef();
        RefUtil.freeRef(inObj);
        throw new IllegalArgumentException(RefString.format("leftDimensions=%s;rightDimensions=%s",
            RefArrays.toString(leftDimensions), RefArrays.toString(rightDimensions)));
      }
    }
    CudaTensorList data = fwd(leftData, rightData.addRef(), leftDimensions, length);
    Accumulator accumulator = new Accumulator(precision, left.getData(), rightData, length, leftDimensions, rightDimensions, left.getAccumulator(), left.isAlive(), right.getAccumulator(), right.isAlive());
    right.freeRef();
    left.freeRef();
    boolean alive = alive(inObj);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("bypassOnError", isBypassOnError());
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
  ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  @NotNull
  private CudaTensorList fwd(TensorList leftData, TensorList rightData, int[] leftDimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              leftDimensions[2], leftDimensions[1], leftDimensions[0],
              leftDimensions[2] * leftDimensions[1] * leftDimensions[0], leftDimensions[1] * leftDimensions[0],
              leftDimensions[0], 1);
          @Nullable final CudaTensor lPtr = gpu.getTensor(leftData.addRef(), precision,
              MemoryType.Device, false);
          @Nullable final CudaTensor rPtr = gpu.getTensor(rightData.addRef(), precision,
              MemoryType.Device, false);
          //assert lPtr.size == rPtr.size;
          @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
              MemoryType.Device, true);
          CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
          CudaMemory rPtrMemory = rPtr.getMemory(gpu.addRef());
          assert rPtrMemory != null;
          assert lPtrMemory != null;
          CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
              lPtr.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0), rPtr.descriptor.getPtr(),
              rPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
          rPtr.freeRef();
          lPtr.freeRef();
          opDescriptor.freeRef();
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          gpu.freeRef();
          lPtrMemory.dirty();
          lPtrMemory.freeRef();
          rPtrMemory.dirty();
          rPtrMemory.freeRef();
          outputPtr.dirty();
          return new CudaTensorList(
              new CudaTensor(outputPtr, outputDescriptor, precision),
              length, leftDimensions, precision);
        }, leftData.addRef(), rightData),
        leftData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList rightData;
    private final int length;
    private final int[] leftDimensions;
    private final int[] rightDimensions;
    private Precision precision;
    private Result.Accumulator leftAccumulator;
    private boolean leftAlive;
    private boolean rightAlive;
    private Result.Accumulator rightAccumulator;
    private @NotNull TensorList leftData;

    public Accumulator(Precision precision, @NotNull TensorList leftData, TensorList rightData, int length, int[] leftDimensions, int[] rightDimensions, Result.Accumulator leftAccumulator, boolean leftAlive, Result.Accumulator rightAccumulator, boolean rightAlive) {
      this.rightData = rightData;
      this.length = length;
      this.leftDimensions = leftDimensions;
      this.rightDimensions = rightDimensions;
      this.precision = precision;
      this.leftAccumulator = leftAccumulator;
      this.leftAlive = leftAlive;
      this.rightAlive = rightAlive;
      this.rightAccumulator = rightAccumulator;
      this.leftData = leftData;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (leftAlive) {
        @Nonnull
        TensorList data = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                      .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                  @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(
                      precision, length, leftDimensions[2], leftDimensions[1], leftDimensions[0],
                      leftDimensions[2] * leftDimensions[1] * leftDimensions[0],
                      leftDimensions[1] * leftDimensions[0], leftDimensions[0], 1);
                  @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                      precision, MemoryType.Device, false);
                  @Nullable final CudaTensor rightTensor = gpu.getTensor(rightData.addRef(), precision,
                      MemoryType.Device, false);
                  //assert deltaTensor.size == rightTensor.size;
                  @Nonnull final CudaMemory outputPtr = gpu.allocate(
                      (long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
                  CudaMemory rightTensorMemory = rightTensor.getMemory(gpu.addRef());
                  assert rightTensorMemory != null;
                  assert deltaTensorMemory != null;
                  CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                      precision.getPointer(1.0), rightTensor.descriptor.getPtr(),
                      rightTensorMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(),
                      outputPtr.getPtr()));
                  rightTensor.freeRef();
                  deltaTensor.freeRef();
                  opDescriptor.freeRef();
                  deltaTensorMemory.dirty();
                  deltaTensorMemory.freeRef();
                  rightTensorMemory.dirty();
                  rightTensorMemory.freeRef();
                  outputPtr.dirty();
                  gpu.freeRef();
                  return new CudaTensorList(
                      new CudaTensor(outputPtr, outputDescriptor, precision),
                      length, leftDimensions, precision);
                }, delta == null ? null : delta.addRef(), rightData.addRef()),
                delta == null ? null : delta.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        TensorList delta1 = data == null ? null : data;
        leftAccumulator.accept(buffer1, delta1);
      }
      if (rightAlive) {
        @Nonnull
        TensorList data = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                      .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                  @Nonnull final CudaDevice.CudaTensorDescriptor expandedDescriptor = gpu.newTensorDescriptor(
                      precision, length, leftDimensions[2], leftDimensions[1], leftDimensions[0],
                      leftDimensions[2] * leftDimensions[1] * leftDimensions[0],
                      leftDimensions[1] * leftDimensions[0], leftDimensions[0], 1);
                  @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                      precision, MemoryType.Device, false);
                  @Nullable final CudaTensor leftTensor = gpu.getTensor(leftData.addRef(), precision, MemoryType.Device,
                      false);
                  //assert deltaTensor.size == rightTensor.size;
                  @Nonnull final CudaMemory outputPtr = gpu.allocate(
                      (long) precision.size * expandedDescriptor.nStride * length, MemoryType.Device, true);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
                  CudaMemory leftTensorMemory = leftTensor.getMemory(gpu.addRef());
                  assert leftTensorMemory != null;
                  assert deltaTensorMemory != null;
                  CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                      precision.getPointer(1.0), leftTensor.descriptor.getPtr(), leftTensorMemory.getPtr(),
                      precision.getPointer(0.0), expandedDescriptor.getPtr(), outputPtr.getPtr()));
                  leftTensor.freeRef();
                  deltaTensor.freeRef();
                  opDescriptor.freeRef();
                  deltaTensorMemory.dirty();
                  deltaTensorMemory.freeRef();
                  leftTensorMemory.dirty();
                  leftTensorMemory.freeRef();
                  outputPtr.dirty();
                  if (RefArrays.equals(rightDimensions, leftDimensions) && length == rightData.length()) {
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    gpu.freeRef();
                    outputPtr.dirty();
                    return new CudaTensorList(
                        new CudaTensor(outputPtr, expandedDescriptor, precision),
                        length, rightDimensions, precision);
                  } else {
                    @Nonnull final CudaDevice.CudaTensorDescriptor reducedOutputDescriptor = gpu.newTensorDescriptor(
                        precision, rightData.length(), rightDimensions[2], rightDimensions[1],
                        rightDimensions[0], rightDimensions[2] * rightDimensions[1] * rightDimensions[0],
                        rightDimensions[1] * rightDimensions[0], rightDimensions[0], 1);
                    long size = (long) precision.size * reducedOutputDescriptor.nStride * rightData.length();
                    @Nonnull final CudaMemory reducedOutputPtr = gpu.allocate(size, MemoryType.Managed.ifEnabled(),
                        true);
                    CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu
                        .cudnnCreateReduceTensorDescriptor(cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD,
                            precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
                            cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES,
                            cudnnIndicesType.CUDNN_32BIT_INDICES);

                    @Nonnull final CudaMemory workspacePtr = gpu.allocate(outputPtr.size, MemoryType.Device, true);
                    @Nonnull final CudaMemory indexPtr = gpu.allocate(3, MemoryType.Device, false);

                    //outputPtr.synchronize();
                    gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
                        workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(1.0),
                        expandedDescriptor.getPtr(), outputPtr.getPtr(), precision.getPointer(0.0),
                        reducedOutputDescriptor.getPtr(), reducedOutputPtr.getPtr());
                    gpu.freeRef();
                    indexPtr.freeRef();
                    reduceTensorDescriptor.freeRef();
                    reducedOutputPtr.dirty();
                    workspacePtr.dirty();
                    workspacePtr.freeRef();
                    outputPtr.dirty();
                    expandedDescriptor.freeRef();
                    outputPtr.freeRef();
                    return new CudaTensorList(
                        new CudaTensor(reducedOutputPtr, reducedOutputDescriptor, precision),
                        rightData.length(), rightDimensions, precision);
                  }
                }, rightData.addRef(), leftData.addRef(),
                delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        TensorList delta1 = data == null ? null : data;
        rightAccumulator.accept(buffer1, delta1);
      }
      if (null != delta)
        delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      rightData.freeRef();
      leftData.freeRef();
      leftAccumulator.freeRef();
      rightAccumulator.freeRef();
    }
  }
}
