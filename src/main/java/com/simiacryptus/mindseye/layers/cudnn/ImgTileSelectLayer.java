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
import com.simiacryptus.ref.wrappers.RefString;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public @RefAware
class ImgTileSelectLayer extends LayerBase implements MultiPrecision<ImgTileSelectLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileSelectLayer.class);
  private int positionX;
  private int positionY;

  private int sizeY;
  private int sizeX;
  private @Nonnull
  Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private ImgTileSelectLayer() {
  }

  public ImgTileSelectLayer(int sizeX, int sizeY, final int positionX, final int positionY) {
    this(sizeX, sizeY, positionX, positionY, Precision.Float);
  }

  public ImgTileSelectLayer(int sizeX, int sizeY, final int positionX, final int positionY,
                            @NotNull Precision precision) {
    this.sizeY = sizeY;
    this.sizeX = sizeX;
    this.positionX = positionX;
    this.positionY = positionY;
    this.precision = precision;
  }

  protected ImgTileSelectLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeY = json.get("sizeY").getAsInt();
    sizeX = json.get("sizeX").getAsInt();
    positionX = json.get("positionX").getAsInt();
    positionY = json.get("positionY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer.class);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgTileSelectLayer setPrecision(@Nonnull final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json, rs);
  }

  public static CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions,
                                final int[] outputDimensions, @Nonnull Precision precision, final int positionX, final int positionY,
                                final boolean dirty) {
    @Nonnull final CudaMemory outputPtr = gpu.allocate(
        (long) input.length() * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size,
        MemoryType.Managed.ifEnabled(), dirty);
    CudaTensor temp_24_0001 = copy(gpu, input == null ? null : input,
        inputDimensions, outputDimensions, positionX, positionY, precision, outputPtr == null ? null : outputPtr);
    return temp_24_0001;
  }

  public static void copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions,
                          final int positionX, final int positionY, Precision precision, final CudaTensor output) {
    RefUtil.freeRef(copy(gpu, input == null ? null : input, inputDimensions,
        new int[]{output.descriptor.width, output.descriptor.height, output.descriptor.channels}, positionX,
        positionY, precision, output.getMemory(gpu)));
    if (null != output)
      output.freeRef();
  }

  public static CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions,
                                final int[] outputDimensions, final int positionX, final int positionY, final Precision precision,
                                final CudaMemory outputPtr) {
    final int length = input.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      throw new IllegalArgumentException("inputDimensions.length");
    }
    if (3 != outputDimensions.length) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    int bands = inputDimensions[2];
    if (bands != outputDimensions[2]) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", bands, outputDimensions[2]));
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(inputDimensions, outputDimensions, new int[]{-positionX, -positionY, 0});
    @Nullable final CudaTensor inputTensor = gpu.getTensor(input == null ? null : input, precision, MemoryType.Device, false);
    int sourceOffset = 0;
    int destinationOffset = 0;
    if (positionX < 0) {
      destinationOffset += Math.abs(positionX);
    } else {
      sourceOffset += Math.abs(positionX);
    }
    if (positionY < 0) {
      destinationOffset += outputDimensions[0] * Math.abs((positionY));
    } else {
      sourceOffset += inputTensor.descriptor.hStride * (Math.abs(positionY));
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= Tensor.length(inputDimensions);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(outputDimensions);

    @Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        inputTensor.descriptor.nStride, //
        inputTensor.descriptor.cStride, //
        inputTensor.descriptor.hStride, //
        inputTensor.descriptor.wStride);
    CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
    if (null != inputTensor)
      inputTensor.freeRef();
    if (RefArrays.equals(viewDim, outputDimensions)) {
      assert sourceOffset >= 0;
      assert destinationOffset == 0;
      CudaTensor temp_24_0002 = new CudaTensor(
          inputTensorMemory.withByteOffset(sourceOffset * precision.size),
          sourceViewDescriptor == null ? null : sourceViewDescriptor, precision);
      if (null != inputTensorMemory)
        inputTensorMemory.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      return temp_24_0002;
    }

    @Nonnull final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
        outputDimensions[1] * outputDimensions[0], //
        outputDimensions[0], //
        1);
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(1.0),
        destinationViewDescriptor.getPtr(), outputPtr.getPtr().withByteOffset(destinationOffset * precision.size)));
    destinationViewDescriptor.freeRef();
    sourceViewDescriptor.freeRef();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    RefUtil.freeRef(outputPtr.dirty());
    RefUtil.freeRef(inputTensorMemory.dirty());
    if (null != inputTensorMemory)
      inputTensorMemory.freeRef();
    @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        outputDimensions[2], //
        outputDimensions[1], //
        outputDimensions[0], //
        outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
        outputDimensions[1] * outputDimensions[0], //
        outputDimensions[0], //
        1);
    CudaTensor temp_24_0003 = new CudaTensor(
        outputPtr == null ? null : outputPtr.addRef(), passbackDescriptor == null ? null : passbackDescriptor,
        precision);
    if (null != outputPtr)
      outputPtr.freeRef();
    return temp_24_0003;
  }

  @Nonnull
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.setAll(viewDim, i -> {
      int value = Math.min(sourceDimensions[i],
          Math.max(0, destinationDimensions[i] - offset[i] - Math.max(-offset[i], 0)));
      if (0 >= value) {
        throw new IllegalArgumentException(RefString.format("%d: src=%d, dst=%d, offset=%d => %d", i, sourceDimensions[i],
            destinationDimensions[i], offset[i], value));
      }
      return value;
    });
    return viewDim;
  }

  public static @SuppressWarnings("unused")
  ImgTileSelectLayer[] addRefs(ImgTileSelectLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRef)
        .toArray((x) -> new ImgTileSelectLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileSelectLayer[][] addRefs(ImgTileSelectLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRefs)
        .toArray((x) -> new ImgTileSelectLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_24_0013 = getCompatibilityLayer();
      Result temp_24_0009 = temp_24_0013
          .eval(Result.addRefs(inObj));
      if (null != temp_24_0013)
        temp_24_0013.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_24_0009;
    }
    assert 1 == inObj.length;
    final Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    inputData.assertAlive();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull
    int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeY && dimIn[1] == sizeX) {
      if (null != inputData)
        inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return input;
    }
    @Nonnull final int[] dimOut = getViewDimensions(dimIn, new int[]{sizeX, sizeY, dimIn[2]},
        new int[]{-positionX, -positionY, 0});
    final ImgTileSelectLayer imgTileSelectLayer = this.addRef();
    final TensorList outputData = CudaSystem.run(RefUtil.wrapInterface(
        (Function<CudnnHandle, CudaTensorList>) gpu -> {
          assert dimOut[0] > 0;
          assert dimOut[1] > 0;
          assert dimOut[2] > 0;
          boolean dirty = dimOut[0] == dimIn[0] && dimOut[1] == dimIn[1];
          return new CudaTensorList(copy(gpu, inputData == null ? null : inputData.addRef(), dimIn, dimOut, precision,
              imgTileSelectLayer.positionX, imgTileSelectLayer.positionY, dirty), length, dimOut, precision);
        }, inputData == null ? null : inputData.addRef(),
        imgTileSelectLayer == null ? null : imgTileSelectLayer.addRef()),
        inputData == null ? null : inputData.addRef());
    if (null != inputData)
      inputData.freeRef();
    int[] outputDimensions = outputData.getDimensions();
    assert length == outputData.length();
    try {
      try {
        try {
          try {
            return new Result(outputData, new Result.Accumulator() {
              {
              }

              @Override
              public void accept(DeltaSet<UUID> buffer, TensorList error) {
                if (!RefArrays.equals(error.getDimensions(), outputDimensions)) {
                  if (null != buffer)
                    buffer.freeRef();
                  AssertionError temp_24_0011 = new AssertionError(
                      RefArrays.toString(error.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
                  if (null != error)
                    error.freeRef();
                  throw temp_24_0011;
                }
                if (error.length() != length) {
                  if (null != buffer)
                    buffer.freeRef();
                  AssertionError temp_24_0012 = new AssertionError(error.length() + " != " + length);
                  if (null != error)
                    error.freeRef();
                  throw temp_24_0012;
                }
                assert error.length() == length;
                if (input.isAlive()) {
                  input.accumulate(buffer == null ? null : buffer.addRef(),
                      CudaSystem.run(RefUtil.wrapInterface(
                          (Function<CudnnHandle, CudaTensorList>) gpu -> {
                            boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
                            final CudaTensor ptr = copy(gpu, error == null ? null : error.addRef(), dimOut, dimIn,
                                precision, -imgTileSelectLayer.positionX, -imgTileSelectLayer.positionY, dirty);
                            CudaTensorList temp_24_0007 = new CudaTensorList(
                                ptr == null ? null : ptr.addRef(), length, dimIn, precision);
                            if (null != ptr)
                              ptr.freeRef();
                            return temp_24_0007;
                          }, error == null ? null : error.addRef(),
                          imgTileSelectLayer == null ? null : imgTileSelectLayer.addRef()),
                          error == null ? null : error.addRef()));
                }
                if (null != error)
                  error.freeRef();
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
                  boolean temp_24_0008 = x.isAlive();
                  if (null != x)
                    x.freeRef();
                  return temp_24_0008;
                });
              }

              public void _free() {
                ReferenceCounting.freeRefs(inObj);
              }
            };
          } finally {
            ReferenceCounting.freeRefs(inObj);
          }
        } finally {
          if (null != outputData)
            outputData.freeRef();
        }
      } finally {
        if (null != imgTileSelectLayer)
          imgTileSelectLayer.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
    json.addProperty("positionX", positionX);
    json.addProperty("positionY", positionY);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  @Override
  public String toString() {
    return "ImgTileSelectLayer{" + "positionX=" + positionX + ", positionY=" + positionY + ", sizeX=" + sizeX
        + ", sizeY=" + sizeY + ", precision=" + precision + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgTileSelectLayer addRef() {
    return (ImgTileSelectLayer) super.addRef();
  }
}
