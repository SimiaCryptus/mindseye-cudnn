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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ProductLayer extends LayerBase implements MultiPrecision<ProductLayer> {

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
  public ProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
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
  public static @SuppressWarnings("unused")
  ProductLayer[] addRefs(@Nullable ProductLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRef).toArray((x) -> new ProductLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ProductLayer[][] addRefs(@Nullable ProductLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRefs)
        .toArray((x) -> new ProductLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_26_0013 = getCompatibilityLayer();
      Result temp_26_0009 = temp_26_0013.eval(Result.addRefs(inObj));
      temp_26_0013.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_26_0009;
    }
    if (inObj.length != 2) {
      IllegalArgumentException temp_26_0010 = new IllegalArgumentException("inObj.length=" + inObj.length);
      ReferenceCounting.freeRefs(inObj);
      throw temp_26_0010;
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
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    if (leftDimensions[0] != rightDimensions[0] && leftDimensions[0] != 1 && 1 != rightDimensions[0] || rightDimensions.length > 1 && leftDimensions[1] != rightDimensions[1] && leftDimensions[1] != 1 && 1 != rightDimensions[1] || rightDimensions.length > 2 && leftDimensions[2] != rightDimensions[2] && leftDimensions[2] != 1 && 1 != rightDimensions[2]) {
      if (isBypassOnError()) {
        RefUtil.freeRef(inObj[1].getData());
        left.freeRef();
        right.freeRef();
        leftData.freeRef();
        rightData.freeRef();
        Result temp_26_0011 = inObj[0];
        ReferenceCounting.freeRefs(inObj);
        return temp_26_0011;
      } else {
        left.freeRef();
        right.freeRef();
        leftData.freeRef();
        rightData.freeRef();
        ReferenceCounting.freeRefs(inObj);
        throw new IllegalArgumentException(RefString.format("leftDimensions=%s;rightDimensions=%s",
            RefArrays.toString(leftDimensions), RefArrays.toString(rightDimensions)));
      }
    }
    try {
      try {
        try {
          try {
            try {
              return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
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
                    CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                    CudaMemory rPtrMemory = rPtr.getMemory(gpu);
                    assert rPtrMemory != null;
                    assert lPtrMemory != null;
                    CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
                        lPtr.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0), rPtr.descriptor.getPtr(),
                        rPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
                    rPtr.freeRef();
                    lPtr.freeRef();
                    opDescriptor.freeRef();
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    RefUtil.freeRef(lPtrMemory.dirty());
                    lPtrMemory.freeRef();
                    RefUtil.freeRef(rPtrMemory.dirty());
                    rPtrMemory.freeRef();
                    RefUtil.freeRef(outputPtr.dirty());
                    CudaTensor cudaTensor = new CudaTensor(outputPtr,
                        outputDescriptor, precision);
                    CudaTensorList temp_26_0005 = new CudaTensorList(cudaTensor.addRef(),
                        length, leftDimensions, precision);
                    cudaTensor.freeRef();
                    return temp_26_0005;
                  }, leftData.addRef(), rightData.addRef()),
                  leftData.addRef()), new Result.Accumulator() {
                {
                }

                @Override
                public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
                  if (left.isAlive()) {
                    @Nonnull
                    TensorList data = CudaSystem
                        .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                              @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                                  .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                              @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(
                                  precision, length, leftDimensions[2], leftDimensions[1], leftDimensions[0],
                                  leftDimensions[2] * leftDimensions[1] * leftDimensions[0],
                                  leftDimensions[1] * leftDimensions[0], leftDimensions[0], 1);
                              @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                                  precision, MemoryType.Device, false);
                              @Nullable final CudaTensor rightTensor = gpu.getTensor(right.getData(), precision,
                                  MemoryType.Device, false);
                              //assert deltaTensor.size == rightTensor.size;
                              @Nonnull final CudaMemory outputPtr = gpu.allocate(
                                  (long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
                              CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                              CudaMemory rightTensorMemory = rightTensor.getMemory(gpu);
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
                              RefUtil.freeRef(deltaTensorMemory.dirty());
                              deltaTensorMemory.freeRef();
                              RefUtil.freeRef(rightTensorMemory.dirty());
                              rightTensorMemory.freeRef();
                              RefUtil.freeRef(outputPtr.dirty());
                              CudaTensor cudaTensor = new CudaTensor(outputPtr,
                                  outputDescriptor, precision);
                              CudaTensorList temp_26_0006 = new CudaTensorList(
                                  cudaTensor.addRef(), length, leftDimensions, precision);
                              cudaTensor.freeRef();
                              return temp_26_0006;
                            }, delta == null ? null : delta.addRef(), right.addRef()),
                            delta == null ? null : delta.addRef());
                    left.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data);
                  }
                  if (right.isAlive()) {
                    @Nonnull
                    TensorList data = CudaSystem
                        .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                              @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                                  .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                              @Nonnull final CudaDevice.CudaTensorDescriptor expandedDescriptor = gpu.newTensorDescriptor(
                                  precision, length, leftDimensions[2], leftDimensions[1], leftDimensions[0],
                                  leftDimensions[2] * leftDimensions[1] * leftDimensions[0],
                                  leftDimensions[1] * leftDimensions[0], leftDimensions[0], 1);
                              @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                                  precision, MemoryType.Device, false);
                              @Nullable final CudaTensor leftTensor = gpu.getTensor(left.getData(), precision, MemoryType.Device,
                                  false);
                              //assert deltaTensor.size == rightTensor.size;
                              @Nonnull final CudaMemory outputPtr = gpu.allocate(
                                  (long) precision.size * expandedDescriptor.nStride * length, MemoryType.Device, true);
                              CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                              CudaMemory leftTensorMemory = leftTensor.getMemory(gpu);
                              assert leftTensorMemory != null;
                              assert deltaTensorMemory != null;
                              CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
                                  deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                                  precision.getPointer(1.0), leftTensor.descriptor.getPtr(), leftTensorMemory.getPtr(),
                                  precision.getPointer(0.0), expandedDescriptor.getPtr(), outputPtr.getPtr()));
                              leftTensor.freeRef();
                              deltaTensor.freeRef();
                              opDescriptor.freeRef();
                              RefUtil.freeRef(deltaTensorMemory.dirty());
                              deltaTensorMemory.freeRef();
                              RefUtil.freeRef(leftTensorMemory.dirty());
                              leftTensorMemory.freeRef();
                              RefUtil.freeRef(outputPtr.dirty());
                              if (RefArrays.equals(rightDimensions, leftDimensions) && length == rightData.length()) {
                                assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                                RefUtil.freeRef(outputPtr.dirty());
                                CudaTensor cudaTensor = new CudaTensor(outputPtr,
                                    expandedDescriptor, precision);
                                CudaTensorList temp_26_0007 = new CudaTensorList(
                                    cudaTensor.addRef(), length, rightDimensions,
                                    precision);
                                cudaTensor.freeRef();
                                return temp_26_0007;
                              } else {
                                @Nonnull final CudaDevice.CudaTensorDescriptor reducedOutputDescriptor = gpu.newTensorDescriptor(
                                    precision, rightData.length(), rightDimensions[2], rightDimensions[1],
                                    rightDimensions[0], rightDimensions[2] * rightDimensions[1] * rightDimensions[0],
                                    rightDimensions[1] * rightDimensions[0], rightDimensions[0], 1);
                                long size = (long) precision.size * reducedOutputDescriptor.nStride
                                    * rightData.length();
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
                                indexPtr.freeRef();
                                reduceTensorDescriptor.freeRef();
                                RefUtil.freeRef(reducedOutputPtr.dirty());
                                RefUtil.freeRef(workspacePtr.dirty());
                                workspacePtr.freeRef();
                                RefUtil.freeRef(outputPtr.dirty());

                                CudaTensor cudaTensor = new CudaTensor(
                                    reducedOutputPtr,
                                    reducedOutputDescriptor, precision);
                                expandedDescriptor.freeRef();
                                outputPtr.freeRef();
                                CudaTensorList temp_26_0008 = new CudaTensorList(
                                    cudaTensor.addRef(), rightData.length(),
                                    rightDimensions, precision);
                                cudaTensor.freeRef();
                                return temp_26_0008;
                              }
                            }, rightData.addRef(), left.addRef(),
                            delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
                    right.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
                }
              }) {

                {
                  Result.addRefs(inObj);
                }

                @Override
                public boolean isAlive() {
                  for (@Nonnull final Result element : inObj)
                    if (element.isAlive()) {
                      return true;
                    }
                  return false;
                }

                @Override
                public void accumulate(@Nullable final DeltaSet<UUID> buffer, @Nullable final TensorList delta) {
                  Result.Accumulator temp_26_0014 = getAccumulator();
                  assert temp_26_0014 != null;
                  temp_26_0014.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                  temp_26_0014.freeRef();
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public void _free() {
                  ReferenceCounting.freeRefs(inObj);
                }

              };
            } finally {
              ReferenceCounting.freeRefs(inObj);
            }
          } finally {
            rightData.freeRef();
          }
        } finally {
          leftData.freeRef();
        }
      } finally {
        right.freeRef();
      }
    } finally {
      left.freeRef();
    }
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
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }
}
