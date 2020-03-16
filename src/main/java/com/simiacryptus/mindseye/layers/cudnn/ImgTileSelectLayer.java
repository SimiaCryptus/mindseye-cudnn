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
import com.simiacryptus.ref.wrappers.RefString;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ImgTileSelectLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgTileSelectLayer.class);
  private int positionX;
  private int positionY;

  private int sizeY;
  private int sizeX;
  private @Nonnull
  Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  private ImgTileSelectLayer() {
  }

  public ImgTileSelectLayer(int sizeX, int sizeY, final int positionX, final int positionY) {
    this(sizeX, sizeY, positionX, positionY, Precision.Float);
  }

  public ImgTileSelectLayer(int sizeX, int sizeY, final int positionX, final int positionY,
                            @Nonnull Precision precision) {
    this.sizeY = sizeY;
    this.sizeX = sizeX;
    this.positionX = positionX;
    this.positionY = positionY;
    this.precision = precision;
  }

  protected ImgTileSelectLayer(@Nonnull final JsonObject json) {
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

  @Nonnull
  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(@Nonnull final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json);
  }

  @Nonnull
  public static CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nonnull final TensorList input, @Nonnull final int[] inputDimensions,
                                @Nonnull final int[] outputDimensions, @Nonnull Precision precision, final int positionX, final int positionY,
                                final boolean dirty) {
    return copy(gpu, input, inputDimensions, outputDimensions, positionX, positionY, precision,
        gpu.allocate(
            (long) input.length() * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size,
            MemoryType.Managed.ifEnabled(), dirty));
  }

  public static void copy(@Nonnull final CudnnHandle gpu, @Nonnull final TensorList input, @Nonnull final int[] inputDimensions,
                          final int positionX, final int positionY, @Nonnull Precision precision, @Nonnull final CudaTensor output) {
    RefUtil.freeRef(copy(
        gpu, input, inputDimensions,
        new int[]{output.descriptor.width, output.descriptor.height, output.descriptor.channels},
        positionX, positionY, precision,
        output.getMemory(gpu.addRef())));
    output.freeRef();
  }

  @Nonnull
  public static CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nonnull final TensorList input, @Nonnull final int[] inputDimensions,
                                @Nonnull final int[] outputDimensions, final int positionX, final int positionY, @Nonnull final Precision precision,
                                @Nullable final CudaMemory outputPtr) {
    final int length = input.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("inputDimensions.length");
    }
    if (3 != outputDimensions.length) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    int bands = inputDimensions[2];
    if (bands != outputDimensions[2]) {
      input.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", bands, outputDimensions[2]));
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(inputDimensions, outputDimensions, new int[]{-positionX, -positionY, 0});
    @Nullable final CudaTensor inputTensor = gpu.getTensor(input, precision, MemoryType.Device, false);
    int sourceOffset = 0;
    int destinationOffset = 0;
    if (positionX < 0) {
      destinationOffset += Math.abs(positionX);
    } else {
      sourceOffset += Math.abs(positionX);
    }
    if (positionY < 0) {
      destinationOffset += outputDimensions[0] * Math.abs(positionY);
    } else {
      sourceOffset += inputTensor.descriptor.hStride * Math.abs(positionY);
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= Tensor.length(inputDimensions);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(outputDimensions);

    final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        inputTensor.descriptor.nStride, //
        inputTensor.descriptor.cStride, //
        inputTensor.descriptor.hStride, //
        inputTensor.descriptor.wStride);
    CudaMemory inputTensorMemory = inputTensor.getMemory(gpu.addRef());
    inputTensor.freeRef();
    if (RefArrays.equals(viewDim, outputDimensions)) {
      assert destinationOffset == 0;
      assert inputTensorMemory != null;
      CudaTensor temp_24_0002 = new CudaTensor(inputTensorMemory.withByteOffset(sourceOffset * precision.size),
          sourceViewDescriptor, precision);
      inputTensorMemory.freeRef();
      if (null != outputPtr)
        outputPtr.freeRef();
      gpu.freeRef();
      return temp_24_0002;
    }

    final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
        outputDimensions[1] * outputDimensions[0], //
        outputDimensions[0], //
        1);
    assert outputPtr != null;
    assert inputTensorMemory != null;
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(1.0),
        destinationViewDescriptor.getPtr(), outputPtr.getPtr().withByteOffset(destinationOffset * precision.size)));
    destinationViewDescriptor.freeRef();
    sourceViewDescriptor.freeRef();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    outputPtr.dirty();
    inputTensorMemory.dirty();
    inputTensorMemory.freeRef();
    final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        outputDimensions[2], //
        outputDimensions[1], //
        outputDimensions[0], //
        outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
        outputDimensions[1] * outputDimensions[0], //
        outputDimensions[0], //
        1);
    gpu.freeRef();
    return new CudaTensor(outputPtr, passbackDescriptor, precision);
  }

  @Nonnull
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.setAll(viewDim, i -> {
      int value = Math.min(sourceDimensions[i],
          Math.max(0, destinationDimensions[i] - offset[i] - Math.max(-offset[i], 0)));
      if (0 >= value) {
        throw new IllegalArgumentException(RefString.format("%d: src=%d, dst=%d, offset=%d => %d", i,
            sourceDimensions[i], destinationDimensions[i], offset[i], value));
      }
      return value;
    });
    return viewDim;
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
    assert 1 == inObj.length;
    final Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    inputData.assertAlive();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull
    int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeY && dimIn[1] == sizeX) {
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    @Nonnull final int[] dimOut = getViewDimensions(dimIn, new int[]{sizeX, sizeY, dimIn[2]},
        new int[]{-positionX, -positionY, 0});
    final TensorList outputData = fwd(inputData, length, dimIn, dimOut);
    int[] outputDimensions = outputData.getDimensions();
    assert length == outputData.length();
    boolean isAlive = RefArrays.stream(inObj).anyMatch(x -> {
      boolean temp_24_0008 = x.isAlive();
      x.freeRef();
      return temp_24_0008;
    });
    Accumulator accumulator = new Accumulator(this.addRef(), outputDimensions, length, dimOut, dimIn, ImgTileSelectLayer.this.precision, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(outputData, accumulator, isAlive);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileSelectLayer addRef() {
    return (ImgTileSelectLayer) super.addRef();
  }

  @NotNull
  private TensorList fwd(TensorList inputData, int length, int[] dimIn, int[] dimOut) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          assert dimOut[0] > 0;
          assert dimOut[1] > 0;
          assert dimOut[2] > 0;
          boolean dirty = dimOut[0] == dimIn[0] && dimOut[1] == dimIn[1];
          return new CudaTensorList(
              copy(gpu, inputData.addRef(), dimIn, dimOut, precision, positionX, positionY, dirty),
              length, dimOut, precision);
        }, inputData.addRef(), this.addRef()),
        inputData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final ImgTileSelectLayer imgTileSelectLayer;
    private final int[] outputDimensions;
    private final int length;
    private final int[] dimOut;
    private final int[] dimIn;
    private Precision precision;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(ImgTileSelectLayer imgTileSelectLayer, int[] outputDimensions, int length, int[] dimOut, int[] dimIn, Precision precision, Result.Accumulator accumulator, boolean alive) {
      this.imgTileSelectLayer = imgTileSelectLayer;
      this.outputDimensions = outputDimensions;
      this.length = length;
      this.dimOut = dimOut;
      this.dimIn = dimIn;
      this.precision = precision;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      if (!RefArrays.equals(error.getDimensions(), outputDimensions)) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_24_0011 = new AssertionError(
            RefArrays.toString(error.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
        error.freeRef();
        throw temp_24_0011;
      }
      if (error.length() != length) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_24_0012 = new AssertionError(error.length() + " != " + length);
        error.freeRef();
        throw temp_24_0012;
      }
      assert error.length() == length;
      if (alive) {
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
              boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
              return new CudaTensorList(
                  copy(gpu, error.addRef(), dimOut, dimIn, precision, -imgTileSelectLayer.positionX, -imgTileSelectLayer.positionY, dirty),
                  length, dimIn, precision);
            }, error.addRef(),
            imgTileSelectLayer.addRef()),
            error.addRef()));
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      imgTileSelectLayer.freeRef();
    }
  }
}
