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
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class SumReducerLayer extends LayerBase
    implements MultiPrecision<SumReducerLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public SumReducerLayer() {
    super();
  }

  protected SumReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SumReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static SumReducerLayer fromJson(@Nonnull final JsonObject json,
                                         com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SumReducerLayer(json);
  }

  public static @SuppressWarnings("unused")
  SumReducerLayer[] addRefs(SumReducerLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayer::addRef)
        .toArray((x) -> new SumReducerLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SumReducerLayer[][] addRefs(SumReducerLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayer::addRefs)
        .toArray((x) -> new SumReducerLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();

    CudaTensorList result = CudaSystem.run(gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      CudaMemory inputMemory = inputTensor.getMemory(gpu);

      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, 1, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull final CudaMemory outputMemory = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      @Nonnull final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      //outputPtr.synchronize();
      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(1.0), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputMemory.getPtr());
      inputMemory.dirty();
      outputMemory.dirty();
      workspacePtr.dirty();
      return new CudaTensorList(new CudaTensor(outputMemory, outputDescriptor, precision), length,
          new int[]{1, 1, 1}, precision);
    });

    return new Result(result, (DeltaSet<UUID> ctx, TensorList delta) -> {
      TensorList passback = new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, length).mapToObj(i -> {
        Tensor tensor = delta.get(i);
        return new Tensor(inputSize).setAll(tensor.get(0));
      }).toArray(i -> new Tensor[i]));
      input.accumulate(ctx, passback);
    }) {
      public void _free() {
        super._free();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SumReducerLayer addRef() {
    return (SumReducerLayer) super.addRef();
  }

}
