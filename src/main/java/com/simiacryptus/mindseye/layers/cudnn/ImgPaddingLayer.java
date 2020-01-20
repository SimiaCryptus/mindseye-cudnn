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
import com.simiacryptus.mindseye.layers.cudnn.ImgCropLayer.Alignment;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgPaddingLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgPaddingLayer.class);
  private ImgCropLayer.Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private int sizeX;
  private int sizeY; // SpatialReflectionPadding
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private ImgPaddingLayer() {
  }

  public ImgPaddingLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  protected ImgPaddingLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    roundUp = json.get("roundUp").getAsBoolean();
    setVerticalAlign(Alignment.valueOf(json.get("verticalAlign").getAsString()));
    setHorizontalAlign(Alignment.valueOf(json.get("horizontalAlign").getAsString()));
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    assert 0 < sizeX;
    assert 0 < sizeY;
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
  public static ImgPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPaddingLayer(json, rs);
  }

  public static void add(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, @Nonnull int[] input_dimensions, @Nonnull int[] output_dimensions,
                         @Nonnull int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input == null ? null : input.addRef(), input_dimensions,
        output_dimensions, offset, length, precision, output_memory == null ? null : output_memory.addRef(), true);
    if (null != output_memory)
      output_memory.freeRef();
    if (null != input)
      input.freeRef();
    if (null == copyParams) {
      return;
    }
    assert copyParams.input_view_descriptor != null;
    if (0 >= copyParams.input_view_descriptor.width) {
      copyParams.freeRef();
      return;
    }
    if (0 >= copyParams.input_view_descriptor.height) {
      copyParams.freeRef();
      return;
    }
    copyParams.add();
    copyParams.freeRef();
  }

  public static void set(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, @Nonnull int[] input_dimensions, @Nonnull int[] output_dimensions,
                         @Nonnull int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input == null ? null : input.addRef(), input_dimensions,
        output_dimensions, offset, length, precision, output_memory == null ? null : output_memory.addRef(), false);
    if (null != output_memory)
      output_memory.freeRef();
    if (null != input)
      input.freeRef();
    if (null == copyParams) {
      return;
    }
    assert copyParams.input_view_descriptor != null;
    if (0 >= copyParams.input_view_descriptor.width) {
      copyParams.freeRef();
      return;
    }
    if (0 >= copyParams.input_view_descriptor.height) {
      copyParams.freeRef();
      return;
    }
    copyParams.set();
    copyParams.freeRef();
  }

  @Nullable
  public static CopyParams getCopyParams(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, int[] input_dimensions,
                                         int[] output_dimensions, int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory,
                                         boolean reflect) {

    int offset_left = offset[0];
    int offset_top = offset[1];

    int input_offset = 0;
    int output_offset = 0;

    int input_channels = input_dimensions[2];
    int input_height = input_dimensions[1];
    int input_width = input_dimensions[0];

    int output_channels = output_dimensions[2];
    int output_height = output_dimensions[1];
    int output_width = output_dimensions[0];

    int view_channels = Math.min(input_channels, output_channels);
    if (input_channels != output_channels) {
      if (null != input)
        input.freeRef();
      if (null != output_memory)
        output_memory.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", input_channels, output_channels));
    }

    assert input != null;
    int input_wStride = input.descriptor.wStride;
    if (input_width < 0) {
      input_width *= -1;
      input_offset += input_wStride * (input_width - 1);
      input_wStride *= -1;
    }
    int output_wStride = 1;
    if (output_width < 0) {
      output_width *= -1;
      output_offset += output_wStride * (output_width - 1);
      output_wStride *= -1;
    }
    int view_width;
    if (offset_left <= 0) {
      offset_left *= -1;
      view_width = Math.min(input_width - offset_left, output_width);
      input_offset += input_wStride * offset_left;
    } else {
      view_width = Math.min(input_width, output_width - offset_left);
      output_offset += output_wStride * offset_left;
    }
    if (view_width <= 0) {
      input.freeRef();
      if (null != output_memory)
        output_memory.freeRef();
      gpu.freeRef();
      return null;
    }

    int input_hStride = input.descriptor.hStride;
    if (input_height < 0) {
      input_height *= -1;
      input_offset += input_hStride * (input_height - 1);
      input_hStride *= -1;
    }
    int output_hStride = output_width;
    if (output_height < 0) {
      output_height *= -1;
      output_offset += output_hStride * (output_height - 1);
      output_hStride *= -1;
    }
    int view_height;
    if (offset_top <= 0) {
      offset_top *= -1;
      view_height = Math.min(input_height - offset_top, output_height);
      input_offset += input_hStride * offset_top;
    } else {
      view_height = Math.min(input_height, output_height - offset_top);
      output_offset += output_hStride * offset_top;
    }
    if (view_height <= 0) {
      input.freeRef();
      if (null != output_memory)
        output_memory.freeRef();
      gpu.freeRef();
      return null;
    }
    assert input_offset >= 0 : input_offset;
    assert output_offset >= 0 : output_offset;
    ImgPaddingLayer.CopyParams temp_05_0017 = new CopyParams(gpu.addRef());
    temp_05_0017.setLength(length);
    ImgPaddingLayer.CopyParams temp_05_0018 = temp_05_0017.addRef();
    temp_05_0018.setPrecision(precision);
    ImgPaddingLayer.CopyParams temp_05_0019 = temp_05_0018.addRef();
    CudaMemory output_memory1 = output_memory == null ? null : output_memory.addRef();
    temp_05_0019.setOutput_memory(output_memory1);
    ImgPaddingLayer.CopyParams temp_05_0020 = temp_05_0019.addRef();
    temp_05_0020.setInput_memory(input.getMemory(gpu.addRef()));
    ImgPaddingLayer.CopyParams temp_05_0021 = temp_05_0020.addRef();
    temp_05_0021.setInput_offset(input_offset);
    ImgPaddingLayer.CopyParams temp_05_0022 = temp_05_0021.addRef();
    temp_05_0022.setOutput_offset(output_offset);
    ImgPaddingLayer.CopyParams temp_05_0023 = temp_05_0022.addRef();
    temp_05_0023.setInput_view_descriptor(gpu.newTensorDescriptor(precision, length, view_channels, view_height, view_width,
          input.descriptor.nStride, input.descriptor.cStride, input_hStride, input_wStride));
    ImgPaddingLayer.CopyParams temp_05_0024 = temp_05_0023.addRef();
    temp_05_0024.setOutput_view_descriptor(gpu.newTensorDescriptor(precision, length, view_channels, view_height, view_width, //
          output_channels * output_height * output_width, //
          output_height * output_width, //
          output_hStride, //
          output_wStride));
    //
    //
    //
    //
    ImgPaddingLayer.CopyParams temp_05_0012 = temp_05_0024.addRef();
    temp_05_0024.freeRef();
    temp_05_0023.freeRef();
    temp_05_0022.freeRef();
    temp_05_0021.freeRef();
    temp_05_0020.freeRef();
    temp_05_0019.freeRef();
    temp_05_0018.freeRef();
    temp_05_0017.freeRef();
    gpu.freeRef();
    if (null != output_memory)
      output_memory.freeRef();
    input.freeRef();
    return temp_05_0012;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgPaddingLayer[] addRefs(@Nullable ImgPaddingLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPaddingLayer::addRef)
        .toArray((x) -> new ImgPaddingLayer[x]);
  }

  public int half(int i, Alignment alignment) {
    if (alignment == Alignment.Left)
      return 0;
    if (alignment == Alignment.Right)
      return i;
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
      Layer temp_05_0025 = getCompatibilityLayer();
      Result temp_05_0013 = temp_05_0025.eval(RefUtil.addRefs(inObj));
      temp_05_0025.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_05_0013;
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
      ReferenceCounting.freeRefs(inObj);
      return input;
    }
    @Nonnull final int[] dimOut = RefArrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      //      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      boolean dirty = false;
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      if (3 != dimIn.length) {
        inputTensor.freeRef();
        throw new IllegalArgumentException("inputDimensions.length");
      }
      if (3 != dimOut.length) {
        inputTensor.freeRef();
        throw new IllegalArgumentException("dimOut.length");
      }
      //log.info(String.format("offset=%d,%d", offsetX, offsetY));
      CudaTensor outputTensor = copy_expand(gpu, inputTensor.addRef(), dimIn, dimOut,
          length, false);
      inputTensor.freeRef();
      CudaTensorList temp_05_0006 = new CudaTensorList(outputTensor == null ? null : outputTensor.addRef(), length,
          dimOut, precision);
      if (null != outputTensor)
        outputTensor.freeRef();
      return temp_05_0006;
    }, inputData.addRef()), inputData.addRef());
    inputData.freeRef();
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (!RefArrays.equals(delta.getDimensions(), output_dimensions)) {
            if (null != buffer)
              buffer.freeRef();
            AssertionError temp_05_0015 = new AssertionError(
                RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(output_dimensions));
            delta.freeRef();
            throw temp_05_0015;
          }
          if (delta.length() != output_length) {
            if (null != buffer)
              buffer.freeRef();
            AssertionError temp_05_0016 = new AssertionError(delta.length() + " != " + output_length);
            delta.freeRef();
            throw temp_05_0016;
          }
          assert delta.length() == length;

          if (input.isAlive()) {
            final TensorList passbackTensorList = CudaSystem
                .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                  @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision,
                      MemoryType.Device, false);
                  CudaTensor backpropTensor = ImgPaddingLayer.this.copy_condense(gpu,
                      errorPtr.addRef(), dimOut, dimIn, length,
                      dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1]);
                  errorPtr.freeRef();
                  CudaTensorList temp_05_0008 = new CudaTensorList(
                      backpropTensor == null ? null : backpropTensor.addRef(), length, dimIn, precision);
                  if (null != backpropTensor)
                    backpropTensor.freeRef();
                  return temp_05_0008;
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
        }
      };
      return new Result(outputData, accumulator) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return RefArrays.stream(RefUtil.addRefs(inObj)).anyMatch(x -> {
            boolean temp_05_0009 = x.isAlive();
            x.freeRef();
            return temp_05_0009;
          });
        }

        @Override
        public void accumulate(@Nullable final DeltaSet<UUID> buffer, @Nullable final TensorList delta) {
          Result.Accumulator temp_05_0026 = getAccumulator();
          assert temp_05_0026 != null;
          temp_05_0026.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          temp_05_0026.freeRef();
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
      input.freeRef();
    }
  }

  @Nullable
  public CudaTensor copy_condense(@Nonnull CudnnHandle gpu, @Nullable CudaTensor inputTensor, @Nonnull int[] dimIn, @Nonnull int[] dimOut, int length,
                                  boolean dirty) {
    if (3 != dimIn.length) {
      if (null != inputTensor)
        inputTensor.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    if (3 != dimOut.length) {
      if (null != inputTensor)
        inputTensor.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimIn.length");
    }
    int offset_left = half(dimOut[0] - dimIn[0], getHorizontalAlign());
    int offset_top = half(dimOut[1] - dimIn[1], getVerticalAlign());
    if (RefArrays.equals(dimIn, dimOut) && offset_left == 0 && offset_top == 0) {
      gpu.freeRef();
      return inputTensor;
    } else {
      CudaMemory output_memory = gpu.allocate((long) length * Tensor.length(dimOut) * precision.size, MemoryType.Device,
          dirty);
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, dimOut, new int[]{offset_left, offset_top},
          length, precision, output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, new int[]{-dimOut[0], dimOut[1], dimOut[2]},
          new int[]{offset_left - dimOut[0], offset_top}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, new int[]{-dimOut[0], dimOut[1], dimOut[2]},
          new int[]{offset_left + dimOut[0], offset_top}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn,
          new int[]{-dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left + dimOut[0], offset_top + dimOut[1]}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn,
          new int[]{-dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left + dimOut[0], offset_top - dimOut[1]}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn,
          new int[]{-dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left - dimOut[0], offset_top + dimOut[1]}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn,
          new int[]{-dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left - dimOut[0], offset_top - dimOut[1]}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, new int[]{dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left, offset_top - dimOut[1]}, length, precision,
          output_memory.addRef());
      add(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, new int[]{dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left, offset_top + dimOut[1]}, length, precision,
          output_memory.addRef());

      CudaTensor temp_05_0010 = new CudaTensor(output_memory.addRef(),
          simpleDescriptor(length, dimOut, gpu), precision);
      output_memory.freeRef();
      if (null != inputTensor)
        inputTensor.freeRef();
      return temp_05_0010;
    }
  }

  @Nullable
  public CudaTensor copy_expand(@Nonnull CudnnHandle gpu, @Nullable CudaTensor inputTensor, @Nonnull int[] dimIn, @Nonnull int[] dimOut, int length,
                                boolean dirty) {
    if (3 != dimOut.length) {
      if (null != inputTensor)
        inputTensor.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    if (3 != dimIn.length) {
      if (null != inputTensor)
        inputTensor.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimIn.length");
    }
    int offset_left = half(dimOut[0] - dimIn[0], getHorizontalAlign());
    int offset_top = half(dimOut[1] - dimIn[1], getVerticalAlign());
    if (RefArrays.equals(dimIn, dimOut) && offset_left == 0 && offset_top == 0) {
      gpu.freeRef();
      return inputTensor;
    } else {
      CudaMemory output_memory = gpu.allocate((long) length * Tensor.length(dimOut) * precision.size, MemoryType.Device,
          dirty);
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), dimIn, dimOut, new int[]{offset_left, offset_top},
          length, precision, output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left - dimIn[0], offset_top}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left + dimIn[0], offset_top}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left + dimIn[0], offset_top + dimIn[1]}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left + dimIn[0], offset_top - dimIn[1]}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left - dimIn[0], offset_top + dimIn[1]}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{-dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left - dimIn[0], offset_top - dimIn[1]}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left, offset_top - dimIn[1]}, length, precision,
          output_memory.addRef());
      set(gpu.addRef(), inputTensor == null ? null : inputTensor.addRef(), new int[]{dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left, offset_top + dimIn[1]}, length, precision,
          output_memory.addRef());

      CudaTensor temp_05_0011 = new CudaTensor(output_memory.addRef(),
          simpleDescriptor(length, dimOut, gpu), precision);
      output_memory.freeRef();
      if (null != inputTensor)
        inputTensor.freeRef();
      return temp_05_0011;
    }
  }

  @Nonnull
  public CudaDevice.CudaTensorDescriptor simpleDescriptor(int length, int[] dimOut, @Nonnull CudnnHandle gpu) {
    CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        dimOut[2], //
        dimOut[1], //
        dimOut[0], //
        dimOut[2] * dimOut[1] * dimOut[0], //
        dimOut[1] * dimOut[0], //
        dimOut[0], //
        1);
    gpu.freeRef();
    return tensorDescriptor;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPaddingLayer addRef() {
    return (ImgPaddingLayer) super.addRef();
  }

  private static class CopyParams extends ReferenceCountingBase {
    public final CudnnHandle gpu;
    public int length;
    public Precision precision;
    public int input_offset;
    public int output_offset;
    @Nullable
    public CudaMemory input_memory;
    @Nullable
    public CudaDevice.CudaTensorDescriptor input_view_descriptor;
    @Nullable
    public CudaMemory output_memory;
    @Nullable
    private CudaDevice.CudaTensorDescriptor output_view_descriptor;

    public CopyParams(CudnnHandle gpu) {
      this.gpu = gpu;
    }

    public void setInput_memory(@Nullable CudaMemory input_memory) {
      CudaMemory temp_05_0001 = input_memory == null ? null : input_memory.addRef();
      if (null != this.input_memory)
        this.input_memory.freeRef();
      this.input_memory = temp_05_0001 == null ? null : temp_05_0001.addRef();
      if (null != temp_05_0001)
        temp_05_0001.freeRef();
      if (null != input_memory)
        input_memory.freeRef();
    }

    public void setInput_offset(int input_offset) {
      this.input_offset = input_offset;
    }

    public void setInput_view_descriptor(@Nullable CudaDevice.CudaTensorDescriptor input_view_descriptor) {
      CudaDevice.CudaTensorDescriptor temp_05_0002 = input_view_descriptor == null ? null
          : input_view_descriptor.addRef();
      if (null != this.input_view_descriptor)
        this.input_view_descriptor.freeRef();
      this.input_view_descriptor = temp_05_0002 == null ? null : temp_05_0002.addRef();
      if (null != temp_05_0002)
        temp_05_0002.freeRef();
      if (null != input_view_descriptor)
        input_view_descriptor.freeRef();
    }

    public void setLength(int length) {
      this.length = length;
    }

    public void setOutput_memory(@Nullable CudaMemory output_memory) {
      CudaMemory temp_05_0003 = output_memory == null ? null : output_memory.addRef();
      if (null != this.output_memory)
        this.output_memory.freeRef();
      this.output_memory = temp_05_0003 == null ? null : temp_05_0003.addRef();
      if (null != temp_05_0003)
        temp_05_0003.freeRef();
      if (null != output_memory)
        output_memory.freeRef();
    }

    public void setOutput_offset(int output_offset) {
      this.output_offset = output_offset;
    }

    public void setOutput_view_descriptor(@Nullable CudaDevice.CudaTensorDescriptor output_view_descriptor) {
      CudaDevice.CudaTensorDescriptor temp_05_0004 = output_view_descriptor == null ? null
          : output_view_descriptor.addRef();
      if (null != this.output_view_descriptor)
        this.output_view_descriptor.freeRef();
      this.output_view_descriptor = temp_05_0004 == null ? null : temp_05_0004.addRef();
      if (null != temp_05_0004)
        temp_05_0004.freeRef();
      if (null != output_view_descriptor)
        output_view_descriptor.freeRef();
    }

    public void setPrecision(Precision precision) {
      this.precision = precision;
    }

    public void set() {
      assert this.input_view_descriptor != null;
      @Nonnull final CudaDevice.CudaTensorDescriptor input_view_descriptor = this.input_view_descriptor.addRef();
      assert output_memory != null;
      CudaMemory output_with_offset = output_memory.withByteOffset(output_offset * precision.size);
      assert input_memory != null;
      CudaMemory input_with_offset = input_memory.withByteOffset(input_offset * precision.size);
      assert output_view_descriptor != null;
      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), input_view_descriptor.getPtr(),
          input_with_offset.getPtr(), precision.getPointer(0.0), output_view_descriptor.getPtr(),
          output_with_offset.getPtr()));
      input_with_offset.freeRef();
      output_with_offset.freeRef();
      input_view_descriptor.freeRef();
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      input_memory.dirty();
      output_memory.dirty();
    }

    public void add() {
      assert this.input_view_descriptor != null;
      @Nonnull final CudaDevice.CudaTensorDescriptor input_view_descriptor = this.input_view_descriptor.addRef();
      assert input_memory != null;
      CudaMemory input_with_offset = input_memory.withByteOffset(input_offset * precision.size);
      assert output_memory != null;
      CudaMemory output_with_offset = output_memory.withByteOffset(output_offset * precision.size);
      assert output_view_descriptor != null;
      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), input_view_descriptor.getPtr(),
          input_with_offset.getPtr(), precision.getPointer(1.0), output_view_descriptor.getPtr(),
          output_with_offset.getPtr()));
      output_with_offset.freeRef();
      input_with_offset.freeRef();
      input_view_descriptor.freeRef();
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      input_memory.dirty();
      output_memory.dirty();
    }

    public void _free() {
      if (null != output_view_descriptor)
        output_view_descriptor.freeRef();
      output_view_descriptor = null;
      if (null != output_memory)
        output_memory.freeRef();
      output_memory = null;
      if (null != input_view_descriptor)
        input_view_descriptor.freeRef();
      input_view_descriptor = null;
      if (null != input_memory)
        input_memory.freeRef();
      input_memory = null;
      if (null != gpu) gpu.freeRef();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    CopyParams addRef() {
      return (CopyParams) super.addRef();
    }
  }
}
