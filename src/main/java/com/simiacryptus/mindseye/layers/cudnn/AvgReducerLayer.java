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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class AvgReducerLayer extends LayerBase implements MultiPrecision {

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

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static AvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgReducerLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nullable final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    assert inObj != null;
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();
    CudaTensorList result = fwd(inputData, length);
    Result.Accumulator accumulator = new Accumulator(length, inputSize, input.getAccumulator());
    input.freeRef();
    return new Result(result, accumulator);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  AvgReducerLayer addRef() {
    return (AvgReducerLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList fwd(TensorList inputData, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());

      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, 1, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull final CudaMemory outputMemory = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      assert inputMemory != null;
      @Nonnull final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      //outputPtr.synchronize();
      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(1.0), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputMemory.getPtr());
      gpu.freeRef();
      indexPtr.freeRef();
      workspacePtr.freeRef();
      reduceTensorDescriptor.freeRef();
      inputTensor.freeRef();
      outputMemory.dirty();
      inputMemory.dirty();
      inputMemory.freeRef();

      return new CudaTensorList(
          new CudaTensor(outputMemory, outputDescriptor, precision),
          length, new int[]{1, 1, 1}, precision);
    }, inputData));
  }

  private static class Accumulator extends Result.Accumulator {

    private final int length;
    private final int[] inputSize;
    private Result.Accumulator accumulator;

    public Accumulator(int length, int[] inputSize, Result.Accumulator accumulator) {
      this.length = length;
      this.inputSize = inputSize;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nonnull TensorList delta) {
      this.accumulator.accept(ctx, new TensorArray(
          RefIntStream.range(0, length).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            Tensor d = delta.get(i);
            double v = d.get(0) / Tensor.length(inputSize);
            d.freeRef();
            Tensor passback = new Tensor(inputSize);
            passback.setAll(v);
            return passback;
          }, delta)).toArray(i -> new Tensor[i])));
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
