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
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class GateBiasLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

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
    boolean alive = Result.anyAlive(inObj);
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
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    CudaTensorList data = fwd(leftData, rightData.addRef(), leftDimensions, length);
    Result.Accumulator accumulator = new Accumulator(rightData, rightDimensions, leftDimensions, length, GateBiasLayer.this.precision, left.getAccumulator(), left.isAlive(), right.isAlive(), right.getAccumulator());
    left.freeRef();
    right.freeRef();
    return new Result(data, accumulator, alive);
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

  @NotNull
  private CudaTensorList fwd(TensorList leftData, TensorList rightData, int[] leftDimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
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
    }, leftData.addRef(), rightData), leftData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList rightData;
    private final int[] rightDimensions;
    private final int[] leftDimensions;
    private final int length;
    private Precision precision;
    private Result.Accumulator leftAccumulator;
    private boolean leftAlive;
    private boolean rightAlive;
    private Result.Accumulator rightAccumulator;

    public Accumulator(TensorList rightData, int[] rightDimensions, int[] leftDimensions, int length, Precision precision, Result.Accumulator leftAccumulator, boolean leftAlive, boolean rightAlive, Result.Accumulator rightAccumulator) {
      this.rightData = rightData;
      this.rightDimensions = rightDimensions;
      this.leftDimensions = leftDimensions;
      this.length = length;
      this.precision = precision;
      this.leftAccumulator = leftAccumulator;
      this.leftAlive = leftAlive;
      this.rightAlive = rightAlive;
      this.rightAccumulator = rightAccumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (leftAlive) {
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        TensorList delta1 = delta == null ? null : delta.addRef();
        leftAccumulator.accept(buffer1, delta1);
      }
      if (rightAlive) {
        @Nonnull
        TensorList data = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, TensorList>) gpu -> {
                  //assert deltaTensor.size == rightTensor.size;
                  if (RefArrays.equals(rightDimensions, leftDimensions) && length == rightData.length()) {
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    assert delta != null;
                    gpu.freeRef();
                    return delta.addRef();
                  } else {
                    final CudaDevice.CudaTensorDescriptor reducedOutputDescriptor = gpu.newTensorDescriptor(
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
                    CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
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
                    gpu.freeRef();
                    indexPtr.freeRef();
                    workspacePtr.freeRef();
                    deltaTensor.freeRef();
                    reduceTensorDescriptor.freeRef();
                    reducedOutputPtr.dirty();
                    deltaTensorMemory.dirty();
                    deltaTensorMemory.freeRef();
                    return new CudaTensorList(
                        new CudaTensor(reducedOutputPtr,
                            reducedOutputDescriptor, precision),
                        rightData.length(), rightDimensions, precision);
                  }
                }, rightData.addRef(), delta == null ? null : delta.addRef()),
                delta == null ? null : delta.addRef());
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
      rightAccumulator.freeRef();
      leftAccumulator.freeRef();
    }
  }
}
