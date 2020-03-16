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

@SuppressWarnings("serial")
public class ImgBandSelectLayer extends LayerBase implements MultiPrecision {

  private int from;
  private int to;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public ImgBandSelectLayer(int from, int to) {
    this.setFrom(from);
    this.setTo(to);
  }

  protected ImgBandSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    setFrom(json.get("from").getAsInt());
    setTo(json.get("to").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer(
        RefIntStream.range(getFrom(), getTo()).toArray());
  }

  public int getFrom() {
    return from;
  }

  public void setFrom(final int from) {
    this.from = from;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public int getTo() {
    return to;
  }

  public void setTo(int to) {
    this.to = to;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert getFrom() < getTo();
    assert getFrom() >= 0;
    assert getTo() > 0;
    assert 1 == inObj.length;
    final Result in0 = inObj[0].addRef();
    if (!CudaSystem.isEnabled()) {
      in0.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    final TensorList inputData = in0.getData();
    assert 3 == inputData.getDimensions().length;
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    @Nonnull final int[] outputDimensions = RefArrays.copyOf(inputDimensions, 3);
    outputDimensions[2] = getTo() - getFrom();
    CudaTensorList data = fwd(inputData, length, outputDimensions);
    Accumulator accumulator = new Accumulator(outputDimensions, length, inputDimensions, ImgBandSelectLayer.this.precision, ImgBandSelectLayer.this.getFrom(), in0.getAccumulator(), in0.isAlive());
    in0.freeRef();
    boolean isAlive = Result.anyAlive(inObj);
    return new Result(data, accumulator, isAlive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("from", getFrom());
    json.addProperty("to", getTo());
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
  ImgBandSelectLayer addRef() {
    return (ImgBandSelectLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList fwd(TensorList inputData, int length, int[] outputDimensions) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor cudaInput = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      final int byteOffset = cudaInput.descriptor.cStride * getFrom() * precision.size;
      final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, length,
          outputDimensions[2], outputDimensions[1], outputDimensions[0], //
          cudaInput.descriptor.nStride, //
          cudaInput.descriptor.cStride, //
          cudaInput.descriptor.hStride, //
          cudaInput.descriptor.wStride);
      CudaMemory cudaInputMemory = cudaInput.getMemory(gpu.addRef());
      cudaInput.freeRef();
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      gpu.freeRef();
      assert cudaInputMemory != null;
      CudaTensor cudaTensor = new CudaTensor(cudaInputMemory.withByteOffset(byteOffset),
          inputDescriptor, precision);
      cudaInputMemory.freeRef();
      CudaTensorList temp_46_0003 = new CudaTensorList(cudaTensor.addRef(), length,
          outputDimensions, precision);
      cudaTensor.freeRef();
      return temp_46_0003;
    }, inputData.addRef()), inputData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] outputDimensions;
    private final int length;
    private final int[] inputDimensions;
    private Precision precision;
    private int from;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(int[] outputDimensions, int length, int[] inputDimensions, Precision precision, int from, Result.Accumulator accumulator, boolean alive) {
      this.outputDimensions = outputDimensions;
      this.length = length;
      this.inputDimensions = inputDimensions;
      this.precision = precision;
      this.from = from;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      int[] deltaDimensions = delta.getDimensions();
      if (!RefArrays.equals(deltaDimensions, outputDimensions)) {
        if (null != buffer)
          buffer.freeRef();
        delta.freeRef();
        throw new AssertionError(
            RefArrays.toString(deltaDimensions) + " != " + RefArrays.toString(outputDimensions));
      }
      if (alive) {
        final TensorList passbackTensorList = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
              final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision,
                  length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
                  inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[0], //
                  1);
              final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision,
                  length, inputDimensions[2], inputDimensions[1], inputDimensions[0], //
                  inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[0], //
                  1);
              final int byteOffset = viewDescriptor.cStride * from
                  * precision.size;
              assert delta.length() == length;
              //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
              @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision,
                  MemoryType.Device, false);
              long size1 = length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0]
                  * precision.size;
              @Nonnull final CudaMemory passbackBuffer = gpu.allocate(size1, MemoryType.Managed.ifEnabled(), false);
              CudaMemory errorPtrMemory = errorPtr.getMemory(gpu.addRef());
              assert errorPtrMemory != null;
              gpu.cudnnTransformTensor(precision.getPointer(1.0), errorPtr.descriptor.getPtr(),
                  errorPtrMemory.getPtr(), precision.getPointer(0.0), viewDescriptor.getPtr(),
                  passbackBuffer.getPtr().withByteOffset(byteOffset));
              gpu.freeRef();
              errorPtr.freeRef();
              viewDescriptor.freeRef();
              errorPtrMemory.dirty();
              errorPtrMemory.freeRef();
              passbackBuffer.dirty();
              return new CudaTensorList(
                  new CudaTensor(passbackBuffer, inputDescriptor, precision),
                  length, inputDimensions, precision);
              //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            }, delta.addRef()), delta);
        this.accumulator.accept(buffer, passbackTensorList);
      } else {
        delta.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
