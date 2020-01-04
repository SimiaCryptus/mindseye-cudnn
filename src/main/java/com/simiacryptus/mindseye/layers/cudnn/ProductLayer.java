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
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ProductLayer extends LayerBase
    implements MultiPrecision<ProductLayer> {

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
    return this;
  }

  public boolean isBypassOnError() {
    return bypassOnError;
  }

  public void setBypassOnError(boolean bypassOnError) {
    this.bypassOnError = bypassOnError;
  }

  @SuppressWarnings("unused")
  public static ProductLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ProductLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    if (inObj.length != 2) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result left = inObj[0];
    Result right = inObj[1];
    final TensorList leftData = left.getData();
    final TensorList rightData = right.getData();
    @Nonnull
    final int[] leftDimensions = leftData.getDimensions();
    @Nonnull
    final int[] rightDimensions = rightData.getDimensions();
    final int length = leftData.length();
    if (3 != leftDimensions.length) {
      throw new IllegalArgumentException(
          "dimensions=" + com.simiacryptus.ref.wrappers.RefArrays.toString(leftDimensions));
    }
    if ((leftDimensions[0] != rightDimensions[0]) && (leftDimensions[0] != 1 && 1 != rightDimensions[0])
        || (leftDimensions.length > 1 && rightDimensions.length > 1) && (leftDimensions[1] != rightDimensions[1])
            && (leftDimensions[1] != 1 && 1 != rightDimensions[1])
        || (leftDimensions.length > 2 && rightDimensions.length > 2) && (leftDimensions[2] != rightDimensions[2])
            && (leftDimensions[2] != 1 && 1 != rightDimensions[2])) {
      if (isBypassOnError()) {
        inObj[1].getData();
        return inObj[0];
      } else {
        throw new IllegalArgumentException(String.format("leftDimensions=%s;rightDimensions=%s",
            com.simiacryptus.ref.wrappers.RefArrays.toString(leftDimensions),
            com.simiacryptus.ref.wrappers.RefArrays.toString(rightDimensions)));
      }
    }
    return new Result(CudaSystem.run(gpu -> {
      @Nonnull
      final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      @Nonnull
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
          leftDimensions[2], leftDimensions[1], leftDimensions[0],
          leftDimensions[2] * leftDimensions[1] * leftDimensions[0], leftDimensions[1] * leftDimensions[0],
          leftDimensions[0], 1);
      @Nullable
      final CudaTensor lPtr = gpu.getTensor(leftData, precision, MemoryType.Device, false);
      @Nullable
      final CudaTensor rPtr = gpu.getTensor(rightData, precision, MemoryType.Device, false);
      //assert lPtr.size == rPtr.size;
      @Nonnull
      final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
          MemoryType.Device, true);
      CudaMemory lPtrMemory = lPtr.getMemory(gpu);
      CudaMemory rPtrMemory = rPtr.getMemory(gpu);
      CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), lPtr.descriptor.getPtr(),
          lPtrMemory.getPtr(), precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      lPtrMemory.dirty();
      rPtrMemory.dirty();
      outputPtr.dirty();
      CudaTensor cudaTensor = new CudaTensor(outputPtr, outputDescriptor, precision);
      return new CudaTensorList(cudaTensor, length, leftDimensions, precision);
    }, leftData), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (left.isAlive()) {
        @Nonnull
        TensorList data = CudaSystem.run(gpu -> {
          @Nonnull
          final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
          @Nonnull
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              leftDimensions[2], leftDimensions[1], leftDimensions[0],
              leftDimensions[2] * leftDimensions[1] * leftDimensions[0], leftDimensions[1] * leftDimensions[0],
              leftDimensions[0], 1);
          @Nullable
          final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
          @Nullable
          final CudaTensor rightTensor = gpu.getTensor(right.getData(), precision, MemoryType.Device, false);
          //assert deltaTensor.size == rightTensor.size;
          @Nonnull
          final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
              MemoryType.Device, true);
          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
          CudaMemory rightTensorMemory = rightTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
              deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(), precision.getPointer(1.0),
              rightTensor.descriptor.getPtr(), rightTensorMemory.getPtr(), precision.getPointer(0.0),
              outputDescriptor.getPtr(), outputPtr.getPtr()));
          deltaTensorMemory.dirty();
          rightTensorMemory.dirty();
          outputPtr.dirty();
          CudaTensor cudaTensor = new CudaTensor(outputPtr, outputDescriptor, precision);
          return new CudaTensorList(cudaTensor, length, leftDimensions, precision);
        }, delta);
        left.accumulate(buffer, data);
      }
      if (right.isAlive()) {
        @Nonnull
        TensorList data = CudaSystem.run(gpu -> {
          @Nonnull
          final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
          @Nonnull
          final CudaDevice.CudaTensorDescriptor expandedDescriptor = gpu.newTensorDescriptor(precision, length,
              leftDimensions[2], leftDimensions[1], leftDimensions[0],
              leftDimensions[2] * leftDimensions[1] * leftDimensions[0], leftDimensions[1] * leftDimensions[0],
              leftDimensions[0], 1);
          @Nullable
          final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
          @Nullable
          final CudaTensor leftTensor = gpu.getTensor(left.getData(), precision, MemoryType.Device, false);
          //assert deltaTensor.size == rightTensor.size;
          @Nonnull
          final CudaMemory outputPtr = gpu.allocate((long) precision.size * expandedDescriptor.nStride * length,
              MemoryType.Device, true);
          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
          CudaMemory leftTensorMemory = leftTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
              deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(), precision.getPointer(1.0),
              leftTensor.descriptor.getPtr(), leftTensorMemory.getPtr(), precision.getPointer(0.0),
              expandedDescriptor.getPtr(), outputPtr.getPtr()));
          deltaTensorMemory.dirty();
          leftTensorMemory.dirty();
          outputPtr.dirty();
          if (com.simiacryptus.ref.wrappers.RefArrays.equals(rightDimensions, leftDimensions)
              && length == rightData.length()) {
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            outputPtr.dirty();
            CudaTensor cudaTensor = new CudaTensor(outputPtr, expandedDescriptor, precision);
            return new CudaTensorList(cudaTensor, length, rightDimensions, precision);
          } else {
            @Nonnull
            final CudaDevice.CudaTensorDescriptor reducedOutputDescriptor = gpu.newTensorDescriptor(precision,
                rightData.length(), rightDimensions[2], rightDimensions[1], rightDimensions[0],
                rightDimensions[2] * rightDimensions[1] * rightDimensions[0], rightDimensions[1] * rightDimensions[0],
                rightDimensions[0], 1);
            long size = (long) precision.size * reducedOutputDescriptor.nStride * rightData.length();
            @Nonnull
            final CudaMemory reducedOutputPtr = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
            CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
                cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code,
                cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES,
                cudnnIndicesType.CUDNN_32BIT_INDICES);

            @Nonnull
            final CudaMemory workspacePtr = gpu.allocate(outputPtr.size, MemoryType.Device, true);
            @Nonnull
            final CudaMemory indexPtr = gpu.allocate(3, MemoryType.Device, false);

            //outputPtr.synchronize();
            gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
                workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(1.0), expandedDescriptor.getPtr(),
                outputPtr.getPtr(), precision.getPointer(0.0), reducedOutputDescriptor.getPtr(),
                reducedOutputPtr.getPtr());
            reducedOutputPtr.dirty();
            workspacePtr.dirty();
            outputPtr.dirty();

            CudaTensor cudaTensor = new CudaTensor(reducedOutputPtr, reducedOutputDescriptor, precision);
            return new CudaTensorList(cudaTensor, rightData.length(), rightDimensions, precision);
          }
        }, delta);
        right.accumulate(buffer, data);
      }
    }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull
        final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }

      @Override
      public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("bypassOnError", isBypassOnError());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ProductLayer[] addRefs(ProductLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRef)
        .toArray((x) -> new ProductLayer[x]);
  }

  public static @SuppressWarnings("unused") ProductLayer[][] addRefs(ProductLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRefs)
        .toArray((x) -> new ProductLayer[x][]);
  }
}
