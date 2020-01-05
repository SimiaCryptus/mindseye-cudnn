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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public @RefAware
class ImgBandSelectLayer extends LayerBase implements MultiPrecision<ImgBandSelectLayer> {

  private int from;
  private int to;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

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

  @Nonnull
  public void setFrom(final int from) {
    this.from = from;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgBandSelectLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public int getTo() {
    return to;
  }

  @Nonnull
  public void setTo(int to) {
    this.to = to;
  }

  @SuppressWarnings("unused")
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgBandSelectLayer[] addRefs(ImgBandSelectLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandSelectLayer::addRef)
        .toArray((x) -> new ImgBandSelectLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgBandSelectLayer[][] addRefs(ImgBandSelectLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandSelectLayer::addRefs)
        .toArray((x) -> new ImgBandSelectLayer[x][]);
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
    TensorList temp_46_0009 = in0.getData();
    assert 3 == temp_46_0009.getDimensions().length;
    if (null != temp_46_0009)
      temp_46_0009.freeRef();
    if (!CudaSystem.isEnabled()) {
      if (null != in0)
        in0.freeRef();
      Layer temp_46_0010 = getCompatibilityLayer();
      Result temp_46_0006 = temp_46_0010
          .eval(Result.addRefs(inObj));
      if (null != temp_46_0010)
        temp_46_0010.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_46_0006;
    }
    final TensorList inputData = in0.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    @Nonnull final int[] outputDimensions = RefArrays.copyOf(inputDimensions, 3);
    outputDimensions[2] = getTo() - getFrom();
    long size = (length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
    try {
      try {
        try {
          return new Result(CudaSystem.run(RefUtil.wrapInterface(
              (Function<CudnnHandle, CudaTensorList>) gpu -> {
                @Nullable final CudaTensor cudaInput = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
                    MemoryType.Device, false);
                final int byteOffset = cudaInput.descriptor.cStride * getFrom() * precision.size;
                @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, length,
                    outputDimensions[2], outputDimensions[1], outputDimensions[0], //
                    cudaInput.descriptor.nStride, //
                    cudaInput.descriptor.cStride, //
                    cudaInput.descriptor.hStride, //
                    cudaInput.descriptor.wStride);
                CudaMemory cudaInputMemory = cudaInput.getMemory(gpu);
                if (null != cudaInput)
                  cudaInput.freeRef();
                assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                CudaTensor cudaTensor = new CudaTensor(cudaInputMemory.withByteOffset(byteOffset),
                    inputDescriptor == null ? null : inputDescriptor, precision);
                if (null != cudaInputMemory)
                  cudaInputMemory.freeRef();
                CudaTensorList temp_46_0003 = new CudaTensorList(
                    cudaTensor == null ? null : cudaTensor.addRef(), length, outputDimensions, precision);
                if (null != cudaTensor)
                  cudaTensor.freeRef();
                return temp_46_0003;
              }, inputData == null ? null : inputData.addRef()), inputData == null ? null : inputData.addRef()),
              new Result.Accumulator() {
                {
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (!RefArrays.equals(delta.getDimensions(), outputDimensions)) {
                    if (null != buffer)
                      buffer.freeRef();
                    AssertionError temp_46_0008 = new AssertionError(
                        RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
                    if (null != delta)
                      delta.freeRef();
                    throw temp_46_0008;
                  }
                  if (in0.isAlive()) {
                    final TensorList passbackTensorList = CudaSystem
                        .run(RefUtil.wrapInterface(
                            (Function<CudnnHandle, CudaTensorList>) gpu -> {
                              @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision,
                                  length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
                                  inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
                                  inputDimensions[1] * inputDimensions[0], //
                                  inputDimensions[0], //
                                  1);
                              @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision,
                                  length, inputDimensions[2], inputDimensions[1], inputDimensions[0], //
                                  inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
                                  inputDimensions[1] * inputDimensions[0], //
                                  inputDimensions[0], //
                                  1);
                              final int byteOffset = viewDescriptor.cStride * ImgBandSelectLayer.this.getFrom()
                                  * precision.size;
                              assert delta.length() == length;
                              //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
                              @Nullable final CudaTensor errorPtr = gpu.getTensor(delta == null ? null : delta.addRef(),
                                  precision, MemoryType.Device, false);
                              long size1 = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0]
                                  * precision.size);
                              @Nonnull final CudaMemory passbackBuffer = gpu.allocate(size1, MemoryType.Managed.ifEnabled(),
                                  false);
                              CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
                              gpu.cudnnTransformTensor(precision.getPointer(1.0), errorPtr.descriptor.getPtr(),
                                  errorPtrMemory.getPtr(), precision.getPointer(0.0), viewDescriptor.getPtr(),
                                  passbackBuffer.getPtr().withByteOffset(byteOffset));
                              if (null != errorPtr)
                                errorPtr.freeRef();
                              viewDescriptor.freeRef();
                              RefUtil.freeRef(errorPtrMemory.dirty());
                              if (null != errorPtrMemory)
                                errorPtrMemory.freeRef();
                              RefUtil.freeRef(passbackBuffer.dirty());
                              CudaTensor cudaTensor = new CudaTensor(passbackBuffer == null ? null : passbackBuffer,
                                  inputDescriptor == null ? null : inputDescriptor, precision);
                              CudaTensorList temp_46_0004 = new CudaTensorList(
                                  cudaTensor == null ? null : cudaTensor.addRef(), length, inputDimensions, precision);
                              if (null != cudaTensor)
                                cudaTensor.freeRef();
                              return temp_46_0004;
                              //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                            }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
                    in0.accumulate(buffer == null ? null : buffer.addRef(),
                        passbackTensorList == null ? null : passbackTensorList.addRef());
                    if (null != passbackTensorList)
                      passbackTensorList.freeRef();
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
                }
              }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                boolean temp_46_0005 = x.isAlive();
                if (null != x)
                  x.freeRef();
                return temp_46_0005;
              });
            }

            @Override
            public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
              Result.Accumulator temp_46_0011 = getAccumulator();
              temp_46_0011.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
              if (null != temp_46_0011)
                temp_46_0011.freeRef();
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public void _free() {
              ReferenceCounting.freeRefs(inObj);
            }
          };
        } finally {
          ReferenceCounting.freeRefs(inObj);
        }
      } finally {
        if (null != inputData)
          inputData.freeRef();
      }
    } finally {
      if (null != in0)
        in0.freeRef();
    }
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
  }

  public @Override
  @SuppressWarnings("unused")
  ImgBandSelectLayer addRef() {
    return (ImgBandSelectLayer) super.addRef();
  }
}
