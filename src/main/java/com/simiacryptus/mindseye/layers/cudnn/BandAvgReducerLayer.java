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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefSystem;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class BandAvgReducerLayer extends LayerBase implements MultiPrecision {

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

  public void setAlpha(double alpha) {
    this.alpha = alpha;
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
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BandAvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandAvgReducerLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nullable final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_32_0005 = getCompatibilityLayer();
      Result temp_32_0004 = temp_32_0005.eval(RefUtil.addRefs(inObj));
      temp_32_0005.freeRef();
      if (null != inObj)
        RefUtil.freeRefs(inObj);
      return temp_32_0004;
    }
    assert inObj != null;
    final Result input = inObj[0].addRef();
    RefUtil.freeRefs(inObj);
    TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();
    if (length <= 0) {
      input.freeRef();
      inputData.freeRef();
      throw new AssertionError();
    }
    if (Tensor.length(inputData.getDimensions()) <= 0) {
      inputData.freeRef();
      return input;
    }
    final int bands = inputSize[2];
    CudaTensorList result = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull final CudaMemory outputPtr = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      CudaMemory inputMemory = inputTensor.getMemory(gpu);
      assert inputMemory != null;
      @Nonnull final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(alpha), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
      indexPtr.freeRef();
      workspacePtr.freeRef();
      reduceTensorDescriptor.freeRef();
      inputTensor.freeRef();
      outputPtr.dirty();
      inputMemory.dirty();

      inputMemory.freeRef();
      CudaTensorList temp_32_0002 = new CudaTensorList(new CudaTensor(outputPtr,
          outputDescriptor, precision), length, new int[]{1, 1, bands}, precision);
      return temp_32_0002;
    }, inputData.addRef()));
    inputData.freeRef();
    int pixels = inputSize[0] * inputSize[1];
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          input.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> ctx, @Nonnull TensorList delta) {
          TensorList passback = new TensorArray(delta.stream().map(x -> {
            final double[] xData = RefArrays.stream(x.getData()).map(v -> v * alpha / pixels).toArray();
            x.freeRef();
            final Tensor tensor = new Tensor(inputSize[0], inputSize[1], inputSize[2]);
            final double[] tensor1Data = tensor.getData();
            for (int p = 0; p < inputSize[0] * inputSize[1]; p++) {
              for (int c = 0; c < inputSize[2]; c++) {
                RefSystem.arraycopy(xData, 0, tensor1Data, p * inputSize[2],
                    inputSize[2]);
              }
            }
            return tensor;
          }).toArray(i -> new Tensor[i]));
          delta.freeRef();
          input.accumulate(ctx == null ? null : ctx.addRef(), passback.addRef());
          if (null != ctx)
            ctx.freeRef();
          passback.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
        }
      };
      return new Result(result, accumulator);
    } finally {
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", alpha);
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
  BandAvgReducerLayer addRef() {
    return (BandAvgReducerLayer) super.addRef();
  }
}
