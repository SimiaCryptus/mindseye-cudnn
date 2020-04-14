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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
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

/**
 * The type Img padding layer.
 */
@SuppressWarnings("serial")
public class ImgPaddingLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgPaddingLayer.class);
  private ImgCropLayer.Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private int sizeX;
  private int sizeY; // SpatialReflectionPadding
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  private ImgPaddingLayer() {
  }

  /**
   * Instantiates a new Img padding layer.
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
   * Instantiates a new Img padding layer.
   *
   * @param json the json
   */
  protected ImgPaddingLayer(@Nonnull final JsonObject json) {
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
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  /**
   * Gets horizontal align.
   *
   * @return the horizontal align
   */
  public Alignment getHorizontalAlign() {
    return horizontalAlign;
  }

  /**
   * Sets horizontal align.
   *
   * @param horizontalAlign the horizontal align
   */
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

  /**
   * Gets vertical align.
   *
   * @return the vertical align
   */
  public Alignment getVerticalAlign() {
    return verticalAlign;
  }

  /**
   * Sets vertical align.
   *
   * @param verticalAlign the vertical align
   */
  public void setVerticalAlign(Alignment verticalAlign) {
    this.verticalAlign = verticalAlign;
  }

  /**
   * Is round up boolean.
   *
   * @return the boolean
   */
  public boolean isRoundUp() {
    return roundUp;
  }

  /**
   * Sets round up.
   *
   * @param roundUp the round up
   */
  public void setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
  }

  /**
   * From json img padding layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img padding layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPaddingLayer(json);
  }

  /**
   * Add.
   *
   * @param gpu               the gpu
   * @param input             the input
   * @param input_dimensions  the input dimensions
   * @param output_dimensions the output dimensions
   * @param offset            the offset
   * @param length            the length
   * @param precision         the precision
   * @param output_memory     the output memory
   */
  public static void add(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, @Nonnull int[] input_dimensions, @Nonnull int[] output_dimensions,
                         @Nonnull int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input, input_dimensions,
        output_dimensions, offset, length, precision, output_memory);
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

  /**
   * Set.
   *
   * @param gpu               the gpu
   * @param input             the input
   * @param input_dimensions  the input dimensions
   * @param output_dimensions the output dimensions
   * @param offset            the offset
   * @param length            the length
   * @param precision         the precision
   * @param output_memory     the output memory
   */
  public static void set(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, @Nonnull int[] input_dimensions, @Nonnull int[] output_dimensions,
                         @Nonnull int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory) {
    CopyParams copyParams = getCopyParams(gpu, input, input_dimensions,
        output_dimensions, offset, length, precision, output_memory);
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

  /**
   * Gets copy params.
   *
   * @param gpu               the gpu
   * @param input             the input
   * @param input_dimensions  the input dimensions
   * @param output_dimensions the output dimensions
   * @param offset            the offset
   * @param length            the length
   * @param precision         the precision
   * @param output_memory     the output memory
   * @return the copy params
   */
  @Nullable
  public static CopyParams getCopyParams(@Nonnull CudnnHandle gpu, @Nullable CudaTensor input, int[] input_dimensions,
                                         int[] output_dimensions, int[] offset, int length, @Nonnull Precision precision, @Nullable CudaMemory output_memory) {

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
    ImgPaddingLayer.CopyParams copyParams = new CopyParams(gpu.addRef());
    copyParams.setLength(length);
    copyParams.setPrecision(precision);
    copyParams.setOutput_memory(output_memory);
    copyParams.setInput_memory(input.getMemory(gpu.addRef()));
    copyParams.setInput_offset(input_offset);
    copyParams.setOutput_offset(output_offset);
    copyParams.setInput_view_descriptor(gpu.newTensorDescriptor(precision, length, view_channels, view_height, view_width,
        input.descriptor.nStride, input.descriptor.cStride, input_hStride, input_wStride));
    copyParams.setOutput_view_descriptor(gpu.newTensorDescriptor(precision, length, view_channels, view_height, view_width, //
        output_channels * output_height * output_width, //
        output_height * output_width, //
        output_hStride, //
        output_wStride));
    gpu.freeRef();
    input.freeRef();
    return copyParams;
  }

  /**
   * Half int.
   *
   * @param i         the
   * @param alignment the alignment
   * @param roundUp   the round up
   * @return the int
   */
  public static int half(int i, Alignment alignment, boolean roundUp) {
    if (alignment == Alignment.Left)
      return 0;
    if (alignment == Alignment.Right)
      return i;
    if (i % 2 == 0)
      return i / 2;
    else if (roundUp)
      return (i + 1) / 2;
    else
      return (i - 1) / 2;
  }

  /**
   * Copy condense cuda tensor.
   *
   * @param gpu             the gpu
   * @param inputTensor     the input tensor
   * @param dimIn           the dim in
   * @param dimOut          the dim out
   * @param length          the length
   * @param dirty           the dirty
   * @param precision       the precision
   * @param horizontalAlign the horizontal align
   * @param verticalAlign   the vertical align
   * @param roundUp         the round up
   * @return the cuda tensor
   */
  @Nullable
  public static CudaTensor copy_condense(@Nonnull CudnnHandle gpu, @Nullable CudaTensor inputTensor, @Nonnull int[] dimIn, @Nonnull int[] dimOut, int length,
                                         boolean dirty, Precision precision, Alignment horizontalAlign, Alignment verticalAlign, boolean roundUp) {
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
    int offset_left = half(dimOut[0] - dimIn[0], horizontalAlign, roundUp);
    int offset_top = half(dimOut[1] - dimIn[1], verticalAlign, roundUp);
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
      add(gpu.addRef(), inputTensor, dimIn, new int[]{dimOut[0], -dimOut[1], dimOut[2]},
          new int[]{offset_left, offset_top + dimOut[1]}, length, precision,
          output_memory.addRef());

      return new CudaTensor(output_memory,
          simpleDescriptor(length, dimOut, gpu, precision), precision);
    }
  }

  /**
   * Simple descriptor cuda device . cuda tensor descriptor.
   *
   * @param length    the length
   * @param dimOut    the dim out
   * @param gpu       the gpu
   * @param precision the precision
   * @return the cuda device . cuda tensor descriptor
   */
  public static CudaDevice.CudaTensorDescriptor simpleDescriptor(int length, int[] dimOut, @Nonnull CudnnHandle gpu, Precision precision) {
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
    Result.Accumulator accumulator = new Accumulator(output_dimensions, output_length, length, dimOut, dimIn, ImgPaddingLayer.this.precision, getHorizontalAlign(), getVerticalAlign(), isRoundUp(), input.getAccumulator(), input.isAlive());
    boolean isAlive = Result.anyAlive(inObj);
    input.freeRef();
    return new Result(outputData, accumulator, isAlive);
  }

  /**
   * Copy expand cuda tensor.
   *
   * @param gpu         the gpu
   * @param inputTensor the input tensor
   * @param dimIn       the dim in
   * @param dimOut      the dim out
   * @param length      the length
   * @param dirty       the dirty
   * @return the cuda tensor
   */
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
    int offset_left = half(dimOut[0] - dimIn[0], getHorizontalAlign(), isRoundUp());
    int offset_top = half(dimOut[1] - dimIn[1], getVerticalAlign(), isRoundUp());
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
      set(gpu.addRef(), inputTensor, new int[]{dimIn[0], -dimIn[1], dimIn[2]}, dimOut,
          new int[]{offset_left, offset_top + dimIn[1]}, length, precision,
          output_memory.addRef());

      return new CudaTensor(output_memory,
          simpleDescriptor(length, dimOut, gpu, precision), precision);
    }
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPaddingLayer addRef() {
    return (ImgPaddingLayer) super.addRef();
  }

  @NotNull
  private TensorList fwd(TensorList inputData, int length, int[] dimIn, int[] dimOut) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, false);
      //      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      boolean dirty = false;
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      if (3 != dimIn.length) {
        inputTensor.freeRef();
        gpu.freeRef();
        throw new IllegalArgumentException("inputDimensions.length");
      }
      if (3 != dimOut.length) {
        inputTensor.freeRef();
        gpu.freeRef();
        throw new IllegalArgumentException("dimOut.length");
      }
      //log.info(String.format("offset=%d,%d", offsetX, offsetY));
      return new CudaTensorList(
          copy_expand(gpu, inputTensor, dimIn, dimOut, length, false),
          length, dimOut, precision);
    }, inputData.addRef()), inputData);
  }

  private static class CopyParams extends ReferenceCountingBase {
    /**
     * The Gpu.
     */
    public final CudnnHandle gpu;
    /**
     * The Length.
     */
    public int length;
    /**
     * The Precision.
     */
    public Precision precision;
    /**
     * The Input offset.
     */
    public int input_offset;
    /**
     * The Output offset.
     */
    public int output_offset;
    /**
     * The Input memory.
     */
    @Nullable
    public CudaMemory input_memory;
    /**
     * The Input view descriptor.
     */
    public CudaDevice.CudaTensorDescriptor input_view_descriptor;
    /**
     * The Output memory.
     */
    @Nullable
    public CudaMemory output_memory;
    private CudaDevice.CudaTensorDescriptor output_view_descriptor;

    /**
     * Instantiates a new Copy params.
     *
     * @param gpu the gpu
     */
    public CopyParams(CudnnHandle gpu) {
      this.gpu = gpu;
    }

    /**
     * Sets input memory.
     *
     * @param input_memory the input memory
     */
    public void setInput_memory(@Nullable CudaMemory input_memory) {
      if (null != this.input_memory)
        this.input_memory.freeRef();
      this.input_memory = input_memory;
    }

    /**
     * Sets input offset.
     *
     * @param input_offset the input offset
     */
    public void setInput_offset(int input_offset) {
      this.input_offset = input_offset;
    }

    /**
     * Sets input view descriptor.
     *
     * @param input_view_descriptor the input view descriptor
     */
    public void setInput_view_descriptor(CudaDevice.CudaTensorDescriptor input_view_descriptor) {
      if (null != this.input_view_descriptor)
        this.input_view_descriptor.freeRef();
      this.input_view_descriptor = input_view_descriptor;
    }

    /**
     * Sets length.
     *
     * @param length the length
     */
    public void setLength(int length) {
      this.length = length;
    }

    /**
     * Sets output memory.
     *
     * @param output_memory the output memory
     */
    public void setOutput_memory(@Nullable CudaMemory output_memory) {
      if (null != this.output_memory)
        this.output_memory.freeRef();
      this.output_memory = output_memory;
    }

    /**
     * Sets output offset.
     *
     * @param output_offset the output offset
     */
    public void setOutput_offset(int output_offset) {
      this.output_offset = output_offset;
    }

    /**
     * Sets output view descriptor.
     *
     * @param output_view_descriptor the output view descriptor
     */
    public void setOutput_view_descriptor(CudaDevice.CudaTensorDescriptor output_view_descriptor) {
      this.output_view_descriptor = output_view_descriptor;
    }

    /**
     * Sets precision.
     *
     * @param precision the precision
     */
    public void setPrecision(Precision precision) {
      this.precision = precision;
    }

    /**
     * Set.
     */
    public void set() {
      assert this.input_view_descriptor != null;
      final CudaDevice.CudaTensorDescriptor input_view_descriptor = this.input_view_descriptor.addRef();
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

    /**
     * Add.
     */
    public void add() {
      assert this.input_view_descriptor != null;
      final CudaDevice.CudaTensorDescriptor input_view_descriptor = this.input_view_descriptor.addRef();
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

  private static class Accumulator extends Result.Accumulator {

    private final int[] output_dimensions;
    private final int output_length;
    private final int length;
    private final int[] dimOut;
    private final int[] dimIn;
    private Precision precision;
    private Alignment horizontalAlign;
    private Alignment verticalAlign;
    private boolean roundUp;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param output_dimensions the output dimensions
     * @param output_length     the output length
     * @param length            the length
     * @param dimOut            the dim out
     * @param dimIn             the dim in
     * @param precision         the precision
     * @param horizontalAlign   the horizontal align
     * @param verticalAlign     the vertical align
     * @param roundUp           the round up
     * @param accumulator       the accumulator
     * @param alive             the alive
     */
    public Accumulator(int[] output_dimensions, int output_length, int length, int[] dimOut, int[] dimIn, Precision precision, Alignment horizontalAlign, Alignment verticalAlign, boolean roundUp, Result.Accumulator accumulator, boolean alive) {
      this.output_dimensions = output_dimensions;
      this.output_length = output_length;
      this.length = length;
      this.dimOut = dimOut;
      this.dimIn = dimIn;
      this.precision = precision;
      this.horizontalAlign = horizontalAlign;
      this.verticalAlign = verticalAlign;
      this.roundUp = roundUp;
      this.accumulator = accumulator;
      this.alive = alive;
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

      if (alive) {
        final TensorList passbackTensorList = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
              @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision,
                  MemoryType.Device, false);
              CudaTensor backpropTensor = copy_condense(gpu,
                  errorPtr, dimOut, dimIn, length,
                  dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1], precision, horizontalAlign, verticalAlign, roundUp);
              return new CudaTensorList(
                  backpropTensor, length, dimIn, precision);
            }, delta.addRef()), delta.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, passbackTensorList);
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
