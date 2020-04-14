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
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Band avg reducer layer.
 */
@SuppressWarnings("serial")
public class BandAvgReducerLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private double alpha = 1.0;

  /**
   * Instantiates a new Band avg reducer layer.
   */
  public BandAvgReducerLayer() {
    super();
  }

  /**
   * Instantiates a new Band avg reducer layer.
   *
   * @param json the json
   */
  protected BandAvgReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   */
  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
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

  /**
   * From json band avg reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the band avg reducer layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static BandAvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandAvgReducerLayer(json);
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
    Accumulator accumulator = new Accumulator(inputSize, alpha, input.getAccumulator());
    input.freeRef();
    return new Result(
        fwd(inputData, inputSize[2], length),
        accumulator
    );
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

  @NotNull
  private CudaTensorList fwd(TensorList inputData, int bands, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull final CudaMemory outputPtr = gpu.allocate(size, MemoryType.Managed.ifEnabled(), true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
          cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
          cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

      CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());
      @Nonnull final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

      try {
        gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
            workspacePtr.size, precision.getPointer(alpha), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
            precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
        outputPtr.dirty();
        inputMemory.dirty();
        return new CudaTensorList(
            new CudaTensor(outputPtr, outputDescriptor, precision),
            length, new int[]{1, 1, bands}, precision);
      } finally {
        gpu.freeRef();
        indexPtr.freeRef();
        workspacePtr.freeRef();
        reduceTensorDescriptor.freeRef();
        inputTensor.freeRef();
        inputMemory.freeRef();
      }
    }, inputData));
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] inputSize;
    private final double alpha;
    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputSize   the input size
     * @param alpha       the alpha
     * @param accumulator the accumulator
     */
    public Accumulator(int[] inputSize, double alpha, Result.Accumulator accumulator) {
      this.inputSize = inputSize;
      this.alpha = alpha;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nonnull TensorList deltaList) {
      TensorList passback = new TensorArray(deltaList.stream().map(delta -> {
        int pixels = inputSize[0] * inputSize[1];
        int bands = inputSize[2];
        assert delta.length() == bands;
        final Tensor tensor = new Tensor(inputSize[0], inputSize[1], bands);
        for (int band = 0; band < bands; band++) {
          int fromIndex = band * pixels;
          tensor.fill(fromIndex, fromIndex + pixels, delta.get(band) * alpha / pixels);
        }
        delta.freeRef();
        return tensor;
      }).toArray(i -> new Tensor[i]));
      deltaList.freeRef();
      Result.Accumulator accumulator = this.accumulator;
      try {
        accumulator.accept(ctx, passback);
      } finally {
        accumulator.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
