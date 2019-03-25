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
import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output png will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgPaddingLayer extends LayerBase implements MultiPrecision<ImgPaddingLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgPaddingLayer.class);
  private Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private int sizeX;
  private int sizeY;
  private Precision precision = Precision.Double;

  /**
   * Instantiates a new Img eval key.
   */
  private ImgPaddingLayer() {
  }

  /**
   * Instantiates a new Img crop key.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgPaddingLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  /**
   * Instantiates a new Img eval key.
   *
   * @param json the json
   * @param rs   the rs
   */
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

  /**
   * From json img eval key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval key
   */
  public static ImgPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPaddingLayer(json, rs);
  }

  public Alignment getVerticalAlign() {
    return verticalAlign;
  }

  public ImgPaddingLayer setVerticalAlign(Alignment verticalAlign) {
    this.verticalAlign = verticalAlign;
    return this;
  }

  public Alignment getHorizontalAlign() {
    return horizontalAlign;
  }

  public ImgPaddingLayer setHorizontalAlign(Alignment horizontalAlign) {
    this.horizontalAlign = horizontalAlign;
    return this;
  }

  public int half(int i, Alignment alignment) {
    if (alignment == Alignment.Left) return 0;
    if (alignment == Alignment.Right) return i;
    if (i % 2 == 0) return i / 2;
    else if (isRoundUp()) return (i + 1) / 2;
    else return (i - 1) / 2;
  }

  /**
   * Gets compatibility key.
   *
   * @return the compatibility key
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      return input;
    }
    @Nonnull final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = CudaSystem.run(gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      inputData.freeRef();
//      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      boolean dirty = false;
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      if (3 != dimIn.length) throw new IllegalArgumentException("inputDimensions.length");
      if (3 != dimOut.length) throw new IllegalArgumentException("dimOut.length");
      //log.info(String.format("offset=%d,%d", offsetX, offsetY));
      CudaTensor outputTensor = copy_expand(gpu, inputTensor, dimIn, dimOut, length, dirty);
      Stream.<ReferenceCounting>of(inputTensor).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(outputTensor, length, dimOut, precision);
    }, inputData);
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    return new Result(outputData, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), output_dimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(output_dimensions));
      }
      if (delta.length() != output_length) {
        throw new AssertionError(delta.length() + " != " + output_length);
      }
      assert delta.length() == length;

      if (input.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          @Nullable final CudaTensor errorPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
          delta.freeRef();
          CudaTensor backpropTensor = copy_condense(gpu, errorPtr, dimOut, dimIn, length, dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1]);
          Stream.<ReferenceCounting>of(errorPtr).forEach(ReferenceCounting::freeRef);
          return CudaTensorList.wrap(backpropTensor, length, dimIn, precision);
        }, delta);
        input.accumulate(buffer, passbackTensorList);
      } else {
        delta.freeRef();
      }

    }) {

      @Override
      public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }

      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }

  public CudaTensor copy_condense(CudnnHandle gpu, CudaTensor inputTensor, int[] dimIn, int[] dimOut, int length, boolean dirty) {
    if (3 != dimIn.length) throw new IllegalArgumentException("dimOut.length");
    if (3 != dimOut.length) throw new IllegalArgumentException("dimIn.length");
    int offset_left = half(dimOut[0] - dimIn[0], getHorizontalAlign());
    int offset_top = half(dimOut[1] - dimIn[1], getVerticalAlign());
    if (Arrays.equals(dimIn, dimOut) && offset_left == 0 && offset_top == 0) {
      return inputTensor;
    } else {
      CudaMemory output_memory = gpu.allocate((long) length * Tensor.length(dimOut) * precision.size, MemoryType.Device, dirty);
      copy(gpu, inputTensor, dimIn, dimOut, new int[]{
          offset_left,
          offset_top
      }, length, precision, output_memory);

      if(offset_left <= 0) {
        add(gpu, inputTensor, new int[]{
            -dimIn[0], dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top
        }, length, precision, output_memory);
        add(gpu, inputTensor, new int[]{
            -dimIn[0], dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top
        }, length, precision, output_memory);
      }
      if(offset_left <= 0 && offset_top <= 0) {
        add(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top + dimIn[1]
        }, length, precision, output_memory);
        add(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top - dimIn[1]
        }, length, precision, output_memory);
        add(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top + dimIn[1]
        }, length, precision, output_memory);
        add(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top - dimIn[1]
        }, length, precision, output_memory);
      }
      if(offset_top <= 0) {
        add(gpu, inputTensor, new int[]{
            dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left,
            offset_top - dimIn[1]
        }, length, precision, output_memory);
        add(gpu, inputTensor, new int[]{
            dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left,
            offset_top + dimIn[1]
        }, length, precision, output_memory);
      }

      return CudaTensor.wrap(output_memory, simpleDescriptor(length, dimOut, gpu), precision);
    }
  }

  public CudaTensor copy_expand(CudnnHandle gpu, CudaTensor inputTensor, int[] dimIn, int[] dimOut, int length, boolean dirty) {
    if (3 != dimOut.length) throw new IllegalArgumentException("dimOut.length");
    if (3 != dimIn.length) throw new IllegalArgumentException("dimIn.length");
    int offset_left = half(dimOut[0] - dimIn[0], getHorizontalAlign());
    int offset_top = half(dimOut[1] - dimIn[1], getVerticalAlign());
    if (Arrays.equals(dimIn, dimOut) && offset_left == 0 && offset_top == 0) {
      return inputTensor;
    } else {
      CudaMemory output_memory = gpu.allocate((long) length * Tensor.length(dimOut) * precision.size, MemoryType.Device, dirty);
      copy(gpu, inputTensor, dimIn, dimOut, new int[]{
          offset_left,
          offset_top
      }, length, precision, output_memory);
      if(offset_left >= 0) {
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top
        }, length, precision, output_memory);
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top
        }, length, precision, output_memory);
      }
      if(offset_left >= 0 && offset_top >= 0) {
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top + dimIn[1]
        }, length, precision, output_memory);
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left + dimIn[0],
            offset_top - dimIn[1]
        }, length, precision, output_memory);
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top + dimIn[1]
        }, length, precision, output_memory);
        copy(gpu, inputTensor, new int[]{
            -dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left - dimIn[0],
            offset_top - dimIn[1]
        }, length, precision, output_memory);
      }
      if(offset_top >= 0) {
        copy(gpu, inputTensor, new int[]{
            dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left,
            offset_top - dimIn[1]
        }, length, precision, output_memory);
        copy(gpu, inputTensor, new int[]{
            dimIn[0], -dimIn[1], dimIn[2]
        }, dimOut, new int[]{
            offset_left,
            offset_top + dimIn[1]
        }, length, precision, output_memory);
      }

      return CudaTensor.wrap(output_memory, simpleDescriptor(length, dimOut, gpu), precision);
    }
  }

  public CudaDevice.CudaTensorDescriptor simpleDescriptor(int length, int[] dimOut, CudnnHandle gpu) {
    return gpu.newTensorDescriptor(
        precision,//
        length,//
        dimOut[2],//
        dimOut[1],//
        dimOut[0],//
        dimOut[2] * dimOut[1] * dimOut[0],//
        dimOut[1] * dimOut[0],//
        dimOut[0],//
        1);
  }

  public static void add(CudnnHandle gpu, CudaTensor input, int[] input_dimensions, int[] output_dimensions, int[] offset, int length, Precision precision, CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input, input_dimensions, output_dimensions, offset, length, precision, output_memory);
    if(null == copyParams) return;
    if(0 >= copyParams.view_width) return;
    if(0 >= copyParams.view_height) return;
    copyParams.add();
  }

  public static void copy(CudnnHandle gpu, CudaTensor input, int[] input_dimensions, int[] output_dimensions, int[] offset, int length, Precision precision, CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input, input_dimensions, output_dimensions, offset, length, precision, output_memory);
    if(null == copyParams) return;
    if(0 >= copyParams.view_width) return;
    if(0 >= copyParams.view_height) return;
    copyParams.copy();
  }

  public static CopyParams getCopyParams(CudnnHandle gpu, CudaTensor input, int[] input_dimensions, int[] output_dimensions, int[] offset, int length, Precision precision, CudaMemory output_memory) {
    CudaMemory inputMemory = input.getMemory(gpu);
    CudaDevice.CudaTensorDescriptor input_descriptor = input.descriptor;

    int offset_left = offset[0];
    int offset_top = offset[1];

    int sourceOffset = 0;
    int destinationOffset = 0;

    int input_channels = input_dimensions[2];
    int input_height = input_dimensions[1];
    int input_width = input_dimensions[0];

    int output_channels = output_dimensions[2];
    int output_height = output_dimensions[1];
    int output_width = output_dimensions[0];

    int view_channels = Math.min(input_channels, output_channels);
    if (input_channels != output_channels) {
      throw new IllegalArgumentException(String.format("%d != %d", input_channels, output_channels));
    }

    int wStride = input_descriptor.wStride;
    if (input_width < 0) {
      input_width *= -1;
      sourceOffset += wStride * (input_width-1);
      wStride *= -1;
    }
    int view_width;
    if (offset_left > 0) {
      view_width = Math.min(input_width, output_width - offset_left);
      destinationOffset += offset_left;
    } else {
      offset_left *= -1;
      view_width = Math.min(input_width - offset_left, output_width);
      sourceOffset += wStride * offset_left;
    }
    if(view_width <= 0) return null;

    int hStride = input_descriptor.hStride;
    if (input_height < 0) {
      input_height *= -1;
      sourceOffset += hStride * (input_height-1);
      hStride *= -1;
    }
    int view_height;
    if (offset_top > 0) {
      view_height = Math.min(input_height, output_height - offset_top);
      destinationOffset += output_width * offset_top;
    } else {
      offset_top *= -1;
      view_height = Math.min(input_height - offset_top, output_height);
      sourceOffset += hStride * offset_top;
    }
    if(view_height <= 0) return null;
    assert sourceOffset >= 0 : sourceOffset;
    assert destinationOffset >= 0 : destinationOffset;
    return new CopyParams(gpu)
        .setLength(length)
        .setPrecision(precision)
        .setOutput_memory(output_memory)
        .setInputMemory(inputMemory)
        .setInput_descriptor(gpu.newTensorDescriptor(precision, length,
            input_descriptor.channels, input_descriptor.height, input_descriptor.width,
            input_descriptor.nStride, input_descriptor.cStride, hStride, wStride))
        .setSourceOffset(sourceOffset)
        .setDestinationOffset(destinationOffset)
        .setOutput_width(output_width)
        .setOutput_height(output_height)
        .setOutput_channels(output_channels)
        .setView_width(view_width)
        .setView_height(view_height)
        .setView_channels(view_channels);
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
  public List<double[]> state() {
    return Arrays.asList();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgPaddingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public ImgPaddingLayer setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
    return this;
  }

  public enum Alignment {
    Center("Center"),
    Left("Right"),
    Right("Left");
    private final String inverse;

    Alignment(String other) {
      this.inverse = other;
    }

    public Alignment getInverse() {
      return Alignment.valueOf(inverse);
    }
  }

  private static class CopyParams {
    public final CudnnHandle gpu;
    public int length;
    public Precision precision;
    public int sourceOffset;
    public int destinationOffset;

    public CudaMemory inputMemory;
    public CudaDevice.CudaTensorDescriptor input_descriptor;
    public int view_width;
    public int view_height;
    public int view_channels;

    public CudaMemory output_memory;
    public int output_width;
    public int output_height;
    public int output_channels;


    public CopyParams(CudnnHandle gpu) {
      this.gpu = gpu;
    }

    public CopyParams copy() {
      @Nonnull final CudaDevice.CudaTensorDescriptor input_view_descriptor = get_input_view_descriptor();
      try {
        @Nonnull final CudaDevice.CudaTensorDescriptor output_view_descriptor = get_output_view_descriptor();
        CudaSystem.handle(gpu.cudnnTransformTensor(
            precision.getPointer(1.0),
            input_view_descriptor.getPtr(), inputMemory.getPtr().withByteOffset(sourceOffset * precision.size),
            precision.getPointer(0.0),
            output_view_descriptor.getPtr(), output_memory.withByteOffset(destinationOffset * precision.size).getPtr()
        ));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputMemory.dirty();
        output_memory.dirty();
        Stream.<ReferenceCounting>of(input_view_descriptor, output_view_descriptor).forEach(ReferenceCounting::freeRef);
        return this;
      } finally {
        inputMemory.freeRef();
      }
    }
    public CopyParams add() {
      @Nonnull final CudaDevice.CudaTensorDescriptor input_view_descriptor = get_input_view_descriptor();
      try {
        @Nonnull final CudaDevice.CudaTensorDescriptor output_view_descriptor = get_output_view_descriptor();
        CudaSystem.handle(gpu.cudnnAddTensor(
            precision.getPointer(1.0),
            input_view_descriptor.getPtr(), inputMemory.getPtr().withByteOffset(sourceOffset * precision.size),
            precision.getPointer(0.0),
            output_view_descriptor.getPtr(), output_memory.withByteOffset(destinationOffset * precision.size).getPtr()
        ));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputMemory.dirty();
        output_memory.dirty();
        Stream.<ReferenceCounting>of(input_view_descriptor, output_view_descriptor).forEach(ReferenceCounting::freeRef);
        return this;
      } finally {
        inputMemory.freeRef();
      }
    }

    public CudaDevice.CudaTensorDescriptor get_output_view_descriptor() {
      return gpu.newTensorDescriptor(
          precision,//
          length,//
          view_channels,//
          view_height,//
          view_width,//
          output_channels * output_height * output_width,//
          output_height * output_width,//
          output_width,//
          1);
    }

    public CudaDevice.CudaTensorDescriptor get_input_view_descriptor() {
      return gpu.newTensorDescriptor(
          precision,//
          length,//
          view_channels,//
          view_height,//
          view_width,//
          input_descriptor.nStride,//
          input_descriptor.cStride,//
          input_descriptor.hStride,//
          input_descriptor.wStride);
    }

    public CopyParams setLength(int length) {
      this.length = length;
      return this;
    }

    public CopyParams setPrecision(Precision precision) {
      this.precision = precision;
      return this;
    }

    public CopyParams setOutput_memory(CudaMemory output_memory) {
      this.output_memory = output_memory;
      return this;
    }

    public CopyParams setInputMemory(CudaMemory inputMemory) {
      this.inputMemory = inputMemory;
      return this;
    }

    public CopyParams setInput_descriptor(CudaDevice.CudaTensorDescriptor input_descriptor) {
      this.input_descriptor = input_descriptor;
      return this;
    }

    public CopyParams setSourceOffset(int sourceOffset) {
      this.sourceOffset = sourceOffset;
      return this;
    }

    public CopyParams setDestinationOffset(int destinationOffset) {
      this.destinationOffset = destinationOffset;
      return this;
    }

    public CopyParams setOutput_width(int output_width) {
      this.output_width = output_width;
      return this;
    }

    public CopyParams setOutput_height(int output_height) {
      this.output_height = output_height;
      return this;
    }

    public CopyParams setOutput_channels(int output_channels) {
      this.output_channels = output_channels;
      return this;
    }

    public CopyParams setView_width(int view_width) {
      this.view_width = view_width;
      return this;
    }

    public CopyParams setView_height(int view_height) {
      this.view_height = view_height;
      return this;
    }

    public CopyParams setView_channels(int view_channels) {
      this.view_channels = view_channels;
      return this;
    }

  }
}
