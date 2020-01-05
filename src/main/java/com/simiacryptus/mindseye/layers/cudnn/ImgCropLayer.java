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
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class ImgCropLayer extends LayerBase
    implements MultiPrecision<ImgCropLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgCropLayer.class);
  private Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private double baseValue = 0.0;
  private int sizeX;
  private int sizeY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private ImgCropLayer() {
  }

  public ImgCropLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  protected ImgCropLayer(@Nonnull final JsonObject json,
                         Map<CharSequence, byte[]> rs) {
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

  public ImgCropLayer setHorizontalAlign(Alignment horizontalAlign) {
    this.horizontalAlign = horizontalAlign;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgCropLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public Alignment getVerticalAlign() {
    return verticalAlign;
  }

  public ImgCropLayer setVerticalAlign(Alignment verticalAlign) {
    this.verticalAlign = verticalAlign;
    return this;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public ImgCropLayer setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgCropLayer fromJson(@Nonnull final JsonObject json,
                                      Map<CharSequence, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ImgCropLayer[] addRefs(ImgCropLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayer::addRef)
        .toArray((x) -> new ImgCropLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgCropLayer[][] addRefs(ImgCropLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayer::addRefs)
        .toArray((x) -> new ImgCropLayer[x][]);
  }

  public CudaTensor copy(final CudnnHandle gpu, final CudaTensor input, final int length, final int[] inputDimensions,
                         final int[] outputDimensions, final boolean dirty, Precision precision, Alignment horizontalAlign,
                         Alignment verticalAlign) {
    if (3 != inputDimensions.length)
      throw new IllegalArgumentException("inputDimensions.length");
    if (3 != outputDimensions.length)
      throw new IllegalArgumentException("dimOut.length");
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    int offset_left = half(outputDimensions[0] - inputDimensions[0], horizontalAlign);
    int offset_top = half(outputDimensions[1] - inputDimensions[1], verticalAlign);
    return copy(gpu, input, length, inputDimensions, outputDimensions, dirty, precision, offset_left, offset_top);
  }

  public CudaTensor copy(CudnnHandle gpu, CudaTensor input, int length, int[] inputDimensions, int[] outputDimensions,
                         boolean dirty, Precision precision, int offset_left, int offset_top) {

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
      throw new IllegalArgumentException(String.format("%d != %d", input_channels, output_channels));
    }
    if (offset_left > 0) {
      destinationOffset += offset_left;
    } else {
      sourceOffset += -offset_left;
    }
    if (offset_top > 0) {
      destinationOffset += output_width * offset_top;
    } else {
      sourceOffset += input.descriptor.hStride * -offset_top;
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;

    @Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        view_channels, //
        view_height, //
        view_width, //
        input.descriptor.nStride, //
        input.descriptor.cStride, //
        input.descriptor.hStride, //
        input.descriptor.wStride);
    CudaMemory inputTensorMemory = input.getMemory(gpu);
    {
      if (output_height == view_height && output_width == view_width) {
        assert sourceOffset >= 0;
        assert destinationOffset == 0;
        return new CudaTensor(inputTensorMemory.withByteOffset(sourceOffset * precision.size), sourceViewDescriptor,
            precision);
      }

      @Nonnull final CudaDevice.CudaTensorDescriptor outputViewDescriptor = gpu.newTensorDescriptor(precision, //
          length, //
          view_channels, //
          view_height, //
          view_width, //
          output_channels * output_height * output_width, //
          output_height * output_width, //
          output_width, //
          1);
      @Nonnull final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
          length, //
          output_channels, //
          output_height, //
          output_width, //
          output_channels * output_height * output_width, //
          output_height * output_width, //
          output_width, //
          1);

      @Nonnull final CudaMemory outputBuffer;
      if (baseValue == 0.0) {
        outputBuffer = gpu.allocate((long) length * output_channels * output_height * output_width * precision.size,
            MemoryType.Device, dirty);
      } else {
        Tensor baseView = new Tensor(outputDimensions).setAll(baseValue);
        outputBuffer = gpu.getTensor(new TensorArray(baseView), precision, MemoryType.Device, true).getMemory(gpu);
      }

      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
          inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(0.0),
          outputViewDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(destinationOffset * precision.size)));
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
      return new CudaTensor(outputBuffer, descriptorCudaResource, precision);
    }
  }

  public int half(int i, Alignment alignment) {
    if (alignment == Alignment.Left)
      return 0;
    if (alignment == Alignment.Right)
      return i;
    return half(i);
  }

  public int half(int i) {
    if (i % 2 == 0)
      return i / 2;
    else if (isRoundUp())
      return (i + 1) / 2;
    else
      return (i - 1) / 2;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull
    int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      return input;
    }
    @Nonnull final int[] dimOut = RefArrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = CudaSystem.run(gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      CudaTensor cudaTensor = copy(gpu, inputTensor, length, dimIn, dimOut, dirty, precision, getHorizontalAlign(),
          getVerticalAlign());
      return new CudaTensorList(cudaTensor, length, dimOut, precision);
    }, inputData);
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    return new Result(outputData, new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList delta) {
        if (!RefArrays.equals(delta.getDimensions(), output_dimensions)) {
          throw new AssertionError(RefArrays.toString(delta.getDimensions()) + " != "
              + RefArrays.toString(output_dimensions));
        }
        if (delta.length() != output_length) {
          throw new AssertionError(delta.length() + " != " + output_length);
        }
        assert delta.length() == length;

        if (input.isAlive()) {
          final TensorList passbackTensorList = CudaSystem.run(gpu -> {
            @Nullable final CudaTensor errorPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
            boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
            CudaTensor cudaTensor = ImgCropLayer.this.copy(gpu, errorPtr, length, dimOut, dimIn, dirty, precision, ImgCropLayer.this.getHorizontalAlign(),
                ImgCropLayer.this.getVerticalAlign());
            return new CudaTensorList(cudaTensor, length, dimIn, precision);
          }, delta);
          input.accumulate(buffer, passbackTensorList);
        }

      }
    }) {

      @Override
      public boolean isAlive() {
        return RefArrays.stream(inObj).anyMatch(x -> x.isAlive());
      }

      @Override
      public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
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
  }

  public @Override
  @SuppressWarnings("unused")
  ImgCropLayer addRef() {
    return (ImgCropLayer) super.addRef();
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
}
