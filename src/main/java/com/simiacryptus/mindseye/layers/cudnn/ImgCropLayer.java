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
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgCropLayer extends LayerBase implements MultiPrecision {
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

  @Nonnull
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
  public CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nullable final CudaTensor input, final int length, @Nonnull final int[] inputDimensions,
                         @Nonnull final int[] outputDimensions, final boolean dirty, @Nonnull Precision precision, Alignment horizontalAlign,
                         Alignment verticalAlign) {
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
    int offset_left = half(outputDimensions[0] - inputDimensions[0], horizontalAlign);
    int offset_top = half(outputDimensions[1] - inputDimensions[1], verticalAlign);
    CudaTensor temp_30_0008 = copy(gpu, input == null ? null : input.addRef(), length, inputDimensions,
        outputDimensions, dirty, precision, offset_left, offset_top);
    if (null != input)
      input.freeRef();
    return temp_30_0008;
  }

  @Nullable
  public CudaTensor copy(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, int length, int[] inputDimensions, int[] outputDimensions,
                         boolean dirty, @Nonnull Precision precision, int offset_left, int offset_top) {

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
    @Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
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
      CudaTensor temp_30_0001 = new CudaTensor(inputTensorMemory.withByteOffset(sourceOffset * precision.size),
          sourceViewDescriptor, precision);
      inputTensorMemory.freeRef();
      gpu.freeRef();
      return temp_30_0001;
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

    destinationViewDescriptor.freeRef();
    @Nonnull CudaMemory outputBuffer = null;
    if (baseValue == 0.0) {
      RefUtil.freeRef(outputBuffer);
      outputBuffer = gpu.allocate((long) length * output_channels * output_height * output_width * precision.size,
          MemoryType.Device, dirty);
    } else {
      Tensor temp_30_0013 = new Tensor(outputDimensions);
      temp_30_0013.setAll(baseValue);
      Tensor baseView = temp_30_0013.addRef();
      temp_30_0013.freeRef();
      CudaTensor temp_30_0014 = gpu.getTensor(new TensorArray(baseView.addRef()), precision,
          MemoryType.Device, true);
      RefUtil.freeRef(outputBuffer);
      outputBuffer = temp_30_0014.getMemory(gpu.addRef());
      temp_30_0014.freeRef();
      baseView.freeRef();
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
    CudaTensor temp_30_0002 = new CudaTensor(outputBuffer,
        descriptorCudaResource.addRef(), precision);
    descriptorCudaResource.freeRef();
    return temp_30_0002;
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
    if (!CudaSystem.isEnabled()) {
      Layer temp_30_0015 = getCompatibilityLayer();
      Result temp_30_0009 = temp_30_0015.eval(RefUtil.addRefs(inObj));
      temp_30_0015.freeRef();
      RefUtil.freeRefs(inObj);
      return temp_30_0009;
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
      RefUtil.freeRefs(inObj);
      return input;
    }
    @Nonnull final int[] dimOut = RefArrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      CudaTensor cudaTensor = copy(gpu, inputTensor.addRef(), length, dimIn, dimOut, dirty,
          precision, getHorizontalAlign(), getVerticalAlign());
      inputTensor.freeRef();
      CudaTensorList temp_30_0004 = new CudaTensorList(cudaTensor == null ? null : cudaTensor.addRef(), length, dimOut,
          precision);
      if (null != cudaTensor)
        cudaTensor.freeRef();
      return temp_30_0004;
    }, inputData.addRef()), inputData.addRef());
    inputData.freeRef();
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    try {
      return new Result(outputData, new Result.Accumulator() {
        {
          input.addRef();
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

          if (input.isAlive()) {
            final TensorList passbackTensorList = CudaSystem
                .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision,
                      MemoryType.Device, false);
                  boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
                  CudaTensor cudaTensor = ImgCropLayer.this.copy(gpu, errorPtr.addRef(),
                      length, dimOut, dimIn, dirty, precision, ImgCropLayer.this.getHorizontalAlign(),
                      ImgCropLayer.this.getVerticalAlign());
                  errorPtr.freeRef();
                  CudaTensorList temp_30_0006 = new CudaTensorList(cudaTensor == null ? null : cudaTensor.addRef(),
                      length, dimIn, precision);
                  if (null != cudaTensor)
                    cudaTensor.freeRef();
                  return temp_30_0006;
                }, delta.addRef()), delta.addRef());
            input.accumulate(buffer == null ? null : buffer.addRef(),
                passbackTensorList == null ? null : passbackTensorList.addRef());
            if (null != passbackTensorList)
              passbackTensorList.freeRef();
          }
          delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return RefArrays.stream(RefUtil.addRefs(inObj)).anyMatch(x -> {
            boolean temp_30_0007 = x.isAlive();
            x.freeRef();
            return temp_30_0007;
          });
        }

        @Override
        public void accumulate(@Nullable final DeltaSet<UUID> buffer, @Nullable final TensorList delta) {
          Result.Accumulator temp_30_0016 = getAccumulator();
          assert temp_30_0016 != null;
          temp_30_0016.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          temp_30_0016.freeRef();
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public void _free() {
          RefUtil.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRefs(inObj);
      input.freeRef();
    }
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
