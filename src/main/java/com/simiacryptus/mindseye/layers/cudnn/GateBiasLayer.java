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
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class GateBiasLayer extends LayerBase implements MultiPrecision {

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
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static GateBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GateBiasLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_52_0010 = getCompatibilityLayer();
      Result temp_52_0007 = temp_52_0010.eval(RefUtil.addRefs(inObj));
      temp_52_0010.freeRef();
      RefUtil.freeRefs(inObj);
      return temp_52_0007;
    }
    if (inObj.length != 2) {
      IllegalArgumentException temp_52_0008 = new IllegalArgumentException("inObj.length=" + inObj.length);
      RefUtil.freeRefs(inObj);
      throw temp_52_0008;
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
      RefUtil.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    try {
      return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
            @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
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
            lPtrMemory.dirty();
            lPtrMemory.freeRef();
            rPtrMemory.dirty();
            rPtrMemory.freeRef();
            outputPtr.dirty();
            CudaTensor cudaTensor = new CudaTensor(outputPtr,
                outputDescriptor, precision);
            CudaTensorList temp_52_0005 = new CudaTensorList(cudaTensor.addRef(),
                length, leftDimensions, precision);
            cudaTensor.freeRef();
            return temp_52_0005;
          }, leftData.addRef(), rightData.addRef()),
          leftData.addRef()), new Result.Accumulator() {
        {
          rightData.addRef();
          right.addRef();
          left.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
          if (left.isAlive()) {
            left.accumulate(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          }
          if (right.isAlive()) {
            @Nonnull
            TensorList data = CudaSystem
                .run(RefUtil.wrapInterface((Function<CudnnHandle, TensorList>) gpu -> {
                      //assert deltaTensor.size == rightTensor.size;
                      if (RefArrays.equals(rightDimensions, leftDimensions) && length == rightData.length()) {
                        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                        assert delta != null;
                        return delta.addRef();
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

                        @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                            precision, MemoryType.Device, false);
                        CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                        assert deltaTensorMemory != null;
                        @Nonnull final CudaMemory workspacePtr = gpu.allocate(deltaTensorMemory.size, MemoryType.Device,
                            true);
                        assert delta != null;
                        @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * delta.length(), MemoryType.Device, false);
                        //outputPtr.synchronize();
                        gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
                            workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(1.0),
                            deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                            precision.getPointer(0.0), reducedOutputDescriptor.getPtr(),
                            reducedOutputPtr.getPtr());
                        indexPtr.freeRef();
                        workspacePtr.freeRef();
                        deltaTensor.freeRef();
                        reduceTensorDescriptor.freeRef();
                        reducedOutputPtr.dirty();
                        deltaTensorMemory.dirty();
                        deltaTensorMemory.freeRef();
                        CudaTensorList temp_52_0006 = new CudaTensorList(
                            new CudaTensor(reducedOutputPtr,
                                reducedOutputDescriptor, precision),
                            rightData.length(), rightDimensions, precision);
                        return temp_52_0006;
                      }
                    }, rightData.addRef(), delta == null ? null : delta.addRef()),
                    delta == null ? null : delta.addRef());
            right.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data);
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
          right.freeRef();
          left.freeRef();
        }
      }) {

        {
          RefUtil.addRefs(inObj);
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
        public final void accumulate(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
          Result.Accumulator temp_52_0011 = getAccumulator();
          assert temp_52_0011 != null;
          temp_52_0011.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          temp_52_0011.freeRef();
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public void _free() {
          RefUtil.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRefs(inObj);
      rightData.freeRef();
      leftData.freeRef();
      right.freeRef();
      left.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
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
  GateBiasLayer addRef() {
    return (GateBiasLayer) super.addRef();
  }
}
