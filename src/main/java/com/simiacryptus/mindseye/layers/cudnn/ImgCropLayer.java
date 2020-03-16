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
public class ImgCropLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgCropLayer.class);
  private Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private double baseValue = 0.0;
  private int sizeX;
  private int sizeY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  private ImgCropLayer() {
  }

  public ImgCropLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  protected ImgCropLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    setBaseValue(json.get("baseValue").getAsDouble());
    roundUp = json.get("roundUp").getAsBoolean();
    setVerticalAlign(Alignment.valueOf(json.get("verticalAlign").getAsString()));
    setHorizontalAlign(Alignment.valueOf(json.get("horizontalAlign").getAsString()));
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  public double getBaseValue() {
    return baseValue;
  }

  public void setBaseValue(double baseValue) {
    this.baseValue = baseValue;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  public Alignment getHorizontalAlign() {
    return horizontalAlign;
  }

  public void setHorizontalAlign(Alignment horizontalAlign) {
    this.horizontalAlign = horizontalAlign;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public Alignment getVerticalAlign() {
    return verticalAlign;
  }

  public void setVerticalAlign(Alignment verticalAlign) {
    this.verticalAlign = verticalAlign;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public void setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgCropLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }

  @Nullable
  public static CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nullable final CudaTensor input, final int length, @Nonnull final int[] inputDimensions,
                                @Nonnull final int[] outputDimensions, final boolean dirty, @Nonnull Precision precision, Alignment horizontalAlign,
                                Alignment verticalAlign, boolean roundUp, double baseValue) {
    if (3 != inputDimensions.length) {
      if (null != input)
        input.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("inputDimensions.length");
    }
    if (3 != outputDimensions.length) {
      if (null != input)
        input.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    int offset_left = half(outputDimensions[0] - inputDimensions[0], horizontalAlign, roundUp);
    int offset_top = half(outputDimensions[1] - inputDimensions[1], verticalAlign, roundUp);
    return copy(gpu, input, length, inputDimensions,
        outputDimensions, dirty, precision, offset_left, offset_top, baseValue);
  }

  @Nullable
  public static CudaTensor copy(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, int length, int[] inputDimensions, int[] outputDimensions,
                                boolean dirty, @Nonnull Precision precision, int offset_left, int offset_top, double baseValue) {

    int sourceOffset = 0;
    int destinationOffset = 0;

    int input_channels = inputDimensions[2];
    int input_height = inputDimensions[1];
    int input_width = inputDimensions[0];

    int output_channels = outputDimensions[2];
    int output_height = outputDimensions[1];
    int output_width = outputDimensions[0];

    int view_channels = Math.min(input_channels, output_channels);
    int view_height = Math.min(input_height, output_height);
    int view_width = Math.min(input_width, output_width);
    if (input_channels != output_channels) {
      if (null != input)
        input.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", input_channels, output_channels));
    }
    if (offset_left > 0) {
      destinationOffset += offset_left;
    } else {
      sourceOffset += -offset_left;
    }
    if (offset_top > 0) {
      destinationOffset += output_width * offset_top;
    } else {
      assert input != null;
      sourceOffset += input.descriptor.hStride * -offset_top;
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;

    assert input != null;
    final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        view_channels, //
        view_height, //
        view_width, //
        input.descriptor.nStride, //
        input.descriptor.cStride, //
        input.descriptor.hStride, //
        input.descriptor.wStride);
    CudaMemory inputTensorMemory = input.getMemory(gpu.addRef());
    input.freeRef();
    if (output_height == view_height && output_width == view_width) {
      assert destinationOffset == 0;
      assert inputTensorMemory != null;
      CudaMemory offset = inputTensorMemory.withByteOffset(sourceOffset * precision.size);
      gpu.freeRef();
      inputTensorMemory.freeRef();
      return new CudaTensor(offset, sourceViewDescriptor, precision);
    }

    final CudaDevice.CudaTensorDescriptor outputViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        view_channels, //
        view_height, //
        view_width, //
        output_channels * output_height * output_width, //
        output_height * output_width, //
        output_width, //
        1);
    final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        output_channels, //
        output_height, //
        output_width, //
        output_channels * output_height * output_width, //
        output_height * output_width, //
        output_width, //
        1);

    destinationViewDescriptor.freeRef();
    @Nonnull CudaMemory outputBuffer = null;
    if (baseValue == 0.0) {
      RefUtil.freeRef(outputBuffer);
      outputBuffer = gpu.allocate((long) length * output_channels * output_height * output_width * precision.size,
          MemoryType.Device, dirty);
    } else {
      Tensor baseView = new Tensor(outputDimensions);
      baseView.setAll(baseValue);
      CudaTensor cudaTensor = gpu.getTensor(new TensorArray(baseView), precision,
          MemoryType.Device, true);
      RefUtil.freeRef(outputBuffer);
      outputBuffer = cudaTensor.getMemory(gpu.addRef());
      cudaTensor.freeRef();
    }

    assert outputBuffer != null;
    assert inputTensorMemory != null;
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(0.0),
        outputViewDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(destinationOffset * precision.size)));
    outputViewDescriptor.freeRef();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    inputTensorMemory.dirty();
    outputBuffer.dirty();
    CudaDevice.CudaTensorDescriptor descriptorCudaResource = gpu.newTensorDescriptor(precision, //
        length, //
        output_channels, //
        output_height, //
        output_width, //
        output_channels * output_height * output_width, //
        output_height * output_width, //
        output_width, //
        1);
    gpu.freeRef();
    sourceViewDescriptor.freeRef();
    inputTensorMemory.freeRef();
    return new CudaTensor(outputBuffer, descriptorCudaResource, precision);
  }

  public static int half(int i, Alignment alignment, boolean roundUp) {
    if (alignment == Alignment.Left)
      return 0;
    if (alignment == Alignment.Right)
      return i;
    return half(i, roundUp);
  }

  public static int half(int i, boolean roundUp) {
    if (i % 2 == 0)
      return i / 2;
    else if (roundUp)
      return (i + 1) / 2;
    else
      return (i - 1) / 2;
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
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull
    int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    @Nonnull final int[] dimOut = RefArrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = fwd(inputData, length, dimIn, dimOut);
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    boolean isAlive = RefArrays.stream(inObj).anyMatch(x -> {
      boolean alive = x.isAlive();
      x.freeRef();
      return alive;
    });
    Accumulator accumulator = new Accumulator(output_dimensions, output_length, length, dimOut, dimIn, precision, getHorizontalAlign(), getVerticalAlign(), isRoundUp(), baseValue, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(outputData, accumulator, isAlive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("baseValue", getBaseValue());
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("roundUp", roundUp);
    json.addProperty("horizontalAlign", getHorizontalAlign().toString());
    json.addProperty("verticalAlign", getVerticalAlign().toString());
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
  ImgCropLayer addRef() {
    return (ImgCropLayer) super.addRef();
  }

  @NotNull
  private TensorList fwd(TensorList inputData, int length, int[] dimIn, int[] dimOut) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      return new CudaTensorList(
          copy(gpu, inputTensor, length, dimIn, dimOut, dirty, precision, getHorizontalAlign(), getVerticalAlign(), isRoundUp(), baseValue),
          length, dimOut, precision);
    }, inputData.addRef()), inputData);
  }

  public enum Alignment {
    Center("Center"), Left("Right"), Right("Left");

    private final String inverse;

    Alignment(String other) {
      this.inverse = other;
    }

    public Alignment getInverse() {
      return Alignment.valueOf(inverse);
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] output_dimensions;
    private final int output_length;
    private final int length;
    private final int[] dimOut;
    private final int[] dimIn;
    private Alignment verticalAlign;
    private Alignment horizontalAlign;
    private Precision precision;
    private boolean roundUp;
    private double baseValue;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(int[] output_dimensions, int output_length, int length, int[] dimOut, int[] dimIn, Precision precision, Alignment horizontalAlign, Alignment verticalAlign, boolean roundUp, double baseValue, Result.Accumulator accumulator, boolean alive) {
      this.output_dimensions = output_dimensions;
      this.output_length = output_length;
      this.length = length;
      this.dimOut = dimOut;
      this.dimIn = dimIn;
      this.verticalAlign = verticalAlign;
      this.horizontalAlign = horizontalAlign;
      this.precision = precision;
      this.roundUp = roundUp;
      this.baseValue = baseValue;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (!RefArrays.equals(delta.getDimensions(), output_dimensions)) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_30_0011 = new AssertionError(
            RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(output_dimensions));
        delta.freeRef();
        throw temp_30_0011;
      }
      if (delta.length() != output_length) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_30_0012 = new AssertionError(delta.length() + " != " + output_length);
        delta.freeRef();
        throw temp_30_0012;
      }
      assert delta.length() == length;

      if (alive) {
        this.accumulator.accept(buffer, CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
              @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision,
                  MemoryType.Device, false);
              boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
              CudaTensor cudaTensor = copy(gpu, errorPtr,
                  length, dimOut, dimIn, dirty, precision, horizontalAlign,
                  verticalAlign, roundUp, baseValue);
              return new CudaTensorList(cudaTensor, length, dimIn, precision);
            }, delta.addRef()), delta));
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
