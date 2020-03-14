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
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntConsumer;
import java.util.function.IntUnaryOperator;

import static com.simiacryptus.mindseye.lang.Result.getData;

@SuppressWarnings("serial")
public class ImgConcatLayer extends LayerBase implements MultiPrecision {

  private int maxBands = -1;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public ImgConcatLayer() {
  }

  protected ImgConcatLayer(@Nonnull final JsonObject json) {
    super(json);
    maxBands = json.get("maxBands").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
    this.parallel = json.get("parallel").getAsBoolean();
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class);
  }

  public int getMaxBands() {
    return maxBands;
  }

  public void setMaxBands(int maxBands) {
    this.maxBands = maxBands;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public boolean isParallel() {
    return parallel;
  }

  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgConcatLayer(json);
  }

  @Nonnull
  public static Tensor eval(@Nonnull final RefList<Tensor> featureImage) {
    ImgConcatLayer layer = new ImgConcatLayer();
    Result temp_31_0011 = layer.eval(featureImage.toArray(new Tensor[]{}));
    layer.freeRef();
    featureImage.freeRef();
    assert temp_31_0011 != null;
    TensorList data = getData(temp_31_0011);
    Tensor tensor = data.get(0);
    data.freeRef();
    return tensor;
  }

  @NotNull
  private static TensorList getTensorList(Result result) {
    try {
      return result.getData();
    } finally {
      result.freeRef();
    }
  }

  private static int[] getDimensions(TensorList tensorList) {
    try {
      return tensorList.getDimensions();
    } finally {
      tensorList.freeRef();
    }
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
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    TensorList data0 = inObj[0].getData();
    final int length = data0.length();
    int[] dimensions = getDimensions(data0);
    assert 3 == dimensions.length;
    @Nonnull final int[] outputDimensions = RefArrays.copyOf(dimensions, dimensions.length);
    assert RefArrays.stream(RefUtil.addRef(inObj)).allMatch(x -> {
      TensorList data = getData(x);
      @Nonnull
      int[] d = data.getDimensions();
      boolean temp_31_0002 = 3 == d.length && d[0] == outputDimensions[0] && d[1] == outputDimensions[1]
          && data.length() == length;
      data.freeRef();
      return temp_31_0002;
    });
    outputDimensions[2] = RefArrays.stream(RefUtil.addRef(inObj)).mapToInt(x -> {
      TensorList temp_31_0017 = getData(x);
      int temp_31_0003 = temp_31_0017.getDimensions()[2];
      temp_31_0017.freeRef();
      return temp_31_0003;
    }).sum();
    if (0 < maxBands && outputDimensions[2] > maxBands) {
      outputDimensions[2] = maxBands;
    }
    CudaTensorList data = fwd(outputDimensions, length, RefUtil.addRef(inObj));
    Accumulator accumulator = new Accumulator(outputDimensions, length, precision, maxBands, parallel, RefUtil.addRef(inObj));
    boolean isAlive = RefArrays.stream(inObj).anyMatch(x -> {
      boolean temp_31_0007 = x.isAlive();
      x.freeRef();
      return temp_31_0007;
    });
    return new Result(data, accumulator, isAlive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
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
  ImgConcatLayer addRef() {
    return (ImgConcatLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList fwd(int[] outputDimensions, int length, @Nonnull Result[] inObj) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      final long outputSize = (long) length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0]
          * precision.size;
      @Nonnull final CudaMemory cudaOutput = gpu.allocate(outputSize, MemoryType.Managed.ifEnabled(), true);
      RefIntStream stream = RefIntStream.range(0, inObj.length);
      //if (!CoreSettings.INSTANCE.isConservative() && parallel) stream = stream.parallel();
      stream.forEach(RefUtil.wrapInterface((IntConsumer) i -> {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        final TensorList input = inObj[i].getData();
        @Nonnull final int[] inputDimensions = input.getDimensions();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        int bandOffset = RefIntStream.range(0, i).map(RefUtil.wrapInterface((IntUnaryOperator) j -> {
          TensorList data = inObj[j].getData();
          int dimension = data.getDimensions()[2];
          data.freeRef();
          return dimension;
        }, RefUtil.addRef(inObj))).sum();
        if (maxBands > 0)
          bandOffset = Math.min(bandOffset, maxBands);
        int inputBands = inputDimensions[2];
        if (maxBands > 0)
          inputBands = Math.min(inputBands, maxBands - bandOffset);
        if (inputBands > 0) {
          @Nullable final CudaTensor cudaInput = gpu.getTensor(input.addRef(), precision,
              MemoryType.Device, false);
          assert maxBands <= 0 || inputBands <= maxBands;
          assert inputBands <= inputDimensions[2];
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputBands, outputDimensions[1], outputDimensions[0], //
              outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
              outputDimensions[1] * outputDimensions[0], //
              outputDimensions[0], //
              1);

          final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, length,
              inputBands, inputDimensions[1], inputDimensions[0], //
              cudaInput.descriptor.nStride, //
              cudaInput.descriptor.cStride, //
              cudaInput.descriptor.hStride, //
              cudaInput.descriptor.wStride);

          int byteOffset = outputDescriptor.cStride * bandOffset * precision.size;
          CudaMemory cudaInputMemory = cudaInput.getMemory(gpu.addRef());
          cudaInput.freeRef();
          assert cudaInputMemory != null;
          gpu.cudnnTransformTensor(precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInputMemory.getPtr(),
              precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(byteOffset));
          inputDescriptor.freeRef();
          outputDescriptor.freeRef();
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          cudaInputMemory.dirty();
          cudaInputMemory.freeRef();
          cudaOutput.dirty();
        }
        input.freeRef();
      }, RefUtil.addRef(inObj), cudaOutput.addRef()));
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(precision, length, outputDimensions[2],
          outputDimensions[1], outputDimensions[0]);
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(cudaOutput,
          outDesc, precision), length, outputDimensions, precision);
    }, RefUtil.addRef(inObj)), RefArrays.stream(inObj).map(result -> {
      return getData(result);
    }).toArray());
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] outputDimensions;
    private final int length;
    private final Result[] inObj;
    private boolean parallel;
    private int maxBands;
    private Precision precision;

    public Accumulator(int[] outputDimensions, int length, Precision precision, int maxBands, boolean parallel, Result... inObj) {
      this.outputDimensions = outputDimensions;
      this.length = length;
      this.inObj = inObj;
      this.parallel = parallel;
      this.maxBands = maxBands;
      this.precision = precision;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      assert delta.getDimensions()[0] == outputDimensions[0];
      assert delta.getDimensions()[1] == outputDimensions[1];
      assert delta.getDimensions()[2] == outputDimensions[2];
      if (!RefArrays.equals(delta.getDimensions(), outputDimensions)) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_31_0010 = new AssertionError(
            RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
        delta.freeRef();
        throw temp_31_0010;
      }
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      @Nonnull
      RefIntStream stream = RefIntStream.range(0, inObj.length);
      if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
        stream = stream.parallel();
      stream.forEach(RefUtil.wrapInterface(i -> {
        final Result input = inObj[i].addRef();
        TensorList temp_31_0019 = input.getData();
        int[] inputDimentions = temp_31_0019.getDimensions();
        assert 3 == inputDimentions.length;
        assert delta.length() == temp_31_0019.length();
        temp_31_0019.freeRef();
        assert inputDimentions[0] == outputDimensions[0];
        assert inputDimentions[1] == outputDimensions[1];
        int bandOffset = RefIntStream.range(0, i).map(RefUtil.wrapInterface(j -> {
          return getDimensions(getTensorList(inObj[j].addRef()))[2];
        }, RefUtil.addRef(inObj))).sum();
        int inputBands = maxBands <= 0 ? inputDimentions[2]
            : Math.min(inputDimentions[2], maxBands - bandOffset);
        if (inputBands > 0 && input.isAlive()) {
          assert inputBands <= inputDimentions[2];
          assert inputBands <= outputDimensions[2];
          final TensorList passbackTensorList = CudaSystem
              .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                @Nullable final CudaTensor cudaDelta = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
                CudaMemory cudaDeltaMemory = cudaDelta.getMemory(gpu.addRef());
                if (inputDimentions[2] == inputBands) {
                  final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision,
                      length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                      cudaDelta.descriptor.nStride, //
                      cudaDelta.descriptor.cStride, //
                      cudaDelta.descriptor.hStride, //
                      cudaDelta.descriptor.wStride);
                  int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                  assert cudaDeltaMemory != null;
                  CudaMemory ptr = cudaDeltaMemory.withByteOffset(byteOffset);
                  cudaDelta.freeRef();
                  cudaDeltaMemory.freeRef();
                  gpu.freeRef();
                  return new CudaTensorList(
                      new CudaTensor(ptr, viewDescriptor, precision),
                      length, inputDimentions, precision);
                } else {
                  final CudaDevice.CudaTensorDescriptor passbackTransferDescriptor = gpu.newTensorDescriptor(
                      precision, length, inputBands, inputDimentions[1], inputDimentions[0], //
                      inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                      inputDimentions[1] * inputDimentions[0], //
                      inputDimentions[0], //
                      1);
                  final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                      length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                      inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                      inputDimentions[1] * inputDimentions[0], //
                      inputDimentions[0], //
                      1);
                  final CudaDevice.CudaTensorDescriptor deltaViewDescriptor = gpu.newTensorDescriptor(precision,
                      length, inputBands, inputDimentions[1], inputDimentions[0], //
                      cudaDelta.descriptor.nStride, //
                      cudaDelta.descriptor.cStride, //
                      cudaDelta.descriptor.hStride, //
                      cudaDelta.descriptor.wStride);
                  @Nonnull final CudaMemory cudaBackprop = gpu.allocate(
                      (long) passbackDescriptor.nStride * length * precision.size,
                      MemoryType.Managed.ifEnabled(), inputBands == inputDimentions[2]);
                  int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                  assert cudaDeltaMemory != null;
                  gpu.cudnnTransformTensor(precision.getPointer(1.0), deltaViewDescriptor.getPtr(),
                      cudaDeltaMemory.getPtr().withByteOffset(byteOffset), precision.getPointer(0.0),
                      passbackTransferDescriptor.getPtr(), cudaBackprop.getPtr());
                  gpu.freeRef();
                  deltaViewDescriptor.freeRef();
                  passbackTransferDescriptor.freeRef();
                  cudaBackprop.dirty();
                  cudaDeltaMemory.dirty();
                  cudaDelta.freeRef();
                  cudaDeltaMemory.freeRef();
                  return new CudaTensorList(
                      new CudaTensor(cudaBackprop, passbackDescriptor, precision),
                      length, inputDimentions, precision);
                }
              }, delta.addRef()));
          DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          Result.Accumulator accumulator = input.getAccumulator();
          try {
            accumulator.accept(buffer1, passbackTensorList);
          } finally {
            accumulator.freeRef();
          }
        }
        //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        input.freeRef();
      }, delta, RefUtil.addRef(inObj), buffer));
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
