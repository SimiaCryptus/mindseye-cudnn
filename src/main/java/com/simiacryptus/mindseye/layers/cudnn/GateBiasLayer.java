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
public @com.simiacryptus.ref.lang.RefAware class GateBiasLayer extends LayerBase
    implements MultiPrecision<GateBiasLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public GateBiasLayer() {
  }

  protected GateBiasLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
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
  public GateBiasLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static GateBiasLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new GateBiasLayer(json);
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
    return new Result(CudaSystem.run(gpu -> {
      @Nonnull
      final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
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
        left.accumulate(buffer, delta);
      }
      if (right.isAlive()) {
        @Nonnull
        TensorList data = CudaSystem.run(gpu -> {
          //assert deltaTensor.size == rightTensor.size;
          if (com.simiacryptus.ref.wrappers.RefArrays.equals(rightDimensions, leftDimensions)
              && length == rightData.length()) {
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            return delta;
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

            @Nullable
            final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
            CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
            @Nonnull
            final CudaMemory workspacePtr = gpu.allocate(deltaTensorMemory.size, MemoryType.Device, true);
            @Nonnull
            final CudaMemory indexPtr = gpu.allocate(12 * delta.length(), MemoryType.Device, false);
            //outputPtr.synchronize();
            gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
                workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                deltaTensorMemory.getPtr(), precision.getPointer(0.0), reducedOutputDescriptor.getPtr(),
                reducedOutputPtr.getPtr());
            reducedOutputPtr.dirty();
            deltaTensorMemory.dirty();
            return new CudaTensorList(new CudaTensor(reducedOutputPtr, reducedOutputDescriptor, precision),
                rightData.length(), rightDimensions, precision);
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
      public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
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
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") GateBiasLayer addRef() {
    return (GateBiasLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") GateBiasLayer[] addRefs(GateBiasLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(GateBiasLayer::addRef)
        .toArray((x) -> new GateBiasLayer[x]);
  }

  public static @SuppressWarnings("unused") GateBiasLayer[][] addRefs(GateBiasLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(GateBiasLayer::addRefs)
        .toArray((x) -> new GateBiasLayer[x][]);
  }
}
