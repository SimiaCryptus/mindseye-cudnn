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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class AvgReducerLayer extends LayerBase implements MultiPrecision<AvgReducerLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public AvgReducerLayer() {
    super();
  }

  protected AvgReducerLayer(@Nonnull final JsonObject json) {
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
  public AvgReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static AvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgReducerLayer(json);
  }

  public static @SuppressWarnings("unused") AvgReducerLayer[] addRefs(AvgReducerLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgReducerLayer::addRef)
        .toArray((x) -> new AvgReducerLayer[x]);
  }

  public static @SuppressWarnings("unused") AvgReducerLayer[][] addRefs(AvgReducerLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgReducerLayer::addRefs)
        .toArray((x) -> new AvgReducerLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_48_0007 = getCompatibilityLayer();
      Result temp_48_0004 = temp_48_0007.eval(Result.addRefs(inObj));
      if (null != temp_48_0007)
        temp_48_0007.freeRef();
      if (null != inObj)
        ReferenceCounting.freeRefs(inObj);
      return temp_48_0004;
    }
    final Result input = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    final TensorList inputData = input.getData();
    @Nonnull
    final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();

    CudaTensorList result = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
          MemoryType.Device, false);
      CudaMemory inputMemory = inputTensor.getMemory(gpu);

      @Nonnull
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, 1, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull
      final CudaMemory outputMemory = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      @Nonnull
      final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull
      final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      //outputPtr.synchronize();
      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(1.0), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputMemory.getPtr());
      indexPtr.freeRef();
      workspacePtr.freeRef();
      if (null != reduceTensorDescriptor)
        reduceTensorDescriptor.freeRef();
      if (null != inputTensor)
        inputTensor.freeRef();
      RefUtil.freeRef(outputMemory.dirty());
      RefUtil.freeRef(inputMemory.dirty());

      if (null != inputMemory)
        inputMemory.freeRef();
      CudaTensorList temp_48_0002 = new CudaTensorList(new CudaTensor(outputMemory == null ? null : outputMemory,
          outputDescriptor == null ? null : outputDescriptor, precision), length, new int[] { 1, 1, 1 }, precision);
      return temp_48_0002;
    }, inputData == null ? null : inputData.addRef()));

    if (null != inputData)
      inputData.freeRef();
    try {
      try {
        return new Result(result, new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> ctx, TensorList delta) {
            input.accumulate(ctx == null ? null : ctx.addRef(), new TensorArray(
                RefIntStream.range(0, length).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                  Tensor tensor = delta.get(i);
                  double v = tensor.get(0) / Tensor.length(inputSize);
                  if (null != tensor)
                    tensor.freeRef();
                  Tensor temp_48_0006 = new Tensor(inputSize);
                  Tensor temp_48_0005 = temp_48_0006.setAll(v);
                  if (null != temp_48_0006)
                    temp_48_0006.freeRef();
                  return temp_48_0005;
                }, delta == null ? null : delta.addRef())).toArray(i -> new Tensor[i])));
            if (null != delta)
              delta.freeRef();
            if (null != ctx)
              ctx.freeRef();
          }

          public @SuppressWarnings("unused") void _free() {
          }
        }) {
          public void _free() {
            super._free();
          }
        };
      } finally {
        if (null != result)
          result.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") AvgReducerLayer addRef() {
    return (AvgReducerLayer) super.addRef();
  }

}
