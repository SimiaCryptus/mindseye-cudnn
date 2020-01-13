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
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class BandAvgReducerLayer extends LayerBase implements MultiPrecision<BandAvgReducerLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private double alpha = 1.0;

  public BandAvgReducerLayer() {
    super();
  }

  protected BandAvgReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }

  public double getAlpha() {
    return alpha;
  }

  public BandAvgReducerLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this.addRef();
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
  public BandAvgReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static BandAvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandAvgReducerLayer(json);
  }

  public static @SuppressWarnings("unused") BandAvgReducerLayer[] addRefs(BandAvgReducerLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayer::addRef)
        .toArray((x) -> new BandAvgReducerLayer[x]);
  }

  public static @SuppressWarnings("unused") BandAvgReducerLayer[][] addRefs(BandAvgReducerLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayer::addRefs)
        .toArray((x) -> new BandAvgReducerLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_32_0005 = getCompatibilityLayer();
      Result temp_32_0004 = temp_32_0005.eval(Result.addRefs(inObj));
      if (null != temp_32_0005)
        temp_32_0005.freeRef();
      if (null != inObj)
        ReferenceCounting.freeRefs(inObj);
      return temp_32_0004;
    }
    final Result input = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    TensorList inputData = input.getData();
    @Nonnull
    final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();
    if (length <= 0) {
      if (null != input)
        input.freeRef();
      if (null != inputData)
        inputData.freeRef();
      throw new AssertionError();
    }
    if (Tensor.length(inputData.getDimensions()) <= 0) {
      if (null != inputData)
        inputData.freeRef();
      return input;
    }
    final int bands = inputSize[2];
    CudaTensorList result = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
          MemoryType.Device, false);
      @Nonnull
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull
      final CudaMemory outputPtr = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      CudaMemory inputMemory = inputTensor.getMemory(gpu);
      @Nonnull
      final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull
      final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(alpha), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
      indexPtr.freeRef();
      workspacePtr.freeRef();
      if (null != reduceTensorDescriptor)
        reduceTensorDescriptor.freeRef();
      if (null != inputTensor)
        inputTensor.freeRef();
      RefUtil.freeRef(outputPtr.dirty());
      RefUtil.freeRef(inputMemory.dirty());

      if (null != inputMemory)
        inputMemory.freeRef();
      CudaTensorList temp_32_0002 = new CudaTensorList(new CudaTensor(outputPtr == null ? null : outputPtr,
          outputDescriptor == null ? null : outputDescriptor, precision), length, new int[] { 1, 1, bands }, precision);
      return temp_32_0002;
    }, inputData == null ? null : inputData.addRef()));
    if (null != inputData)
      inputData.freeRef();
    int pixels = inputSize[0] * inputSize[1];
    try {
      try {
        return new Result(result, new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> ctx, TensorList delta) {
            TensorList passback;
            passback = new TensorArray(delta.stream().map(x -> {
              final double[] xData = RefArrays.stream(x.getData()).map(v -> v * alpha / pixels).toArray();
              if (null != x)
                x.freeRef();
              final Tensor tensor = new Tensor(inputSize[0], inputSize[1], inputSize[2]);
              final double[] tensor1Data = tensor.getData();
              for (int p = 0; p < inputSize[0] * inputSize[1]; p++) {
                for (int c = 0; c < inputSize[2]; c++) {
                  com.simiacryptus.ref.wrappers.RefSystem.arraycopy(xData, 0, tensor1Data, p * inputSize[2],
                      inputSize[2]);
                }
              }
              return tensor;
            }).toArray(i -> new Tensor[i]));
            if (null != delta)
              delta.freeRef();
            input.accumulate(ctx == null ? null : ctx.addRef(), passback == null ? null : passback.addRef());
            if (null != ctx)
              ctx.freeRef();
            if (null != passback)
              passback.freeRef();
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
    json.addProperty("alpha", alpha);
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

  public @Override @SuppressWarnings("unused") BandAvgReducerLayer addRef() {
    return (BandAvgReducerLayer) super.addRef();
  }
}
