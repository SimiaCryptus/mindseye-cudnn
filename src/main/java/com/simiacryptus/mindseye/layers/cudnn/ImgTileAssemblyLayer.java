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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class ImgTileAssemblyLayer extends LayerBase
    implements MultiPrecision<ImgTileAssemblyLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileAssemblyLayer.class);

  private int columns;
  private int rows;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel;

  private ImgTileAssemblyLayer() {
  }

  public ImgTileAssemblyLayer(int columns, int rows) {
    this.columns = columns;
    this.rows = rows;
  }

  protected ImgTileAssemblyLayer(@Nonnull final JsonObject json,
                                 Map<CharSequence, byte[]> rs) {
    super(json);
    columns = json.get("columns").getAsInt();
    rows = json.get("rows").getAsInt();
    this.parallel = json.get("parallel").getAsBoolean();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer.class);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgTileAssemblyLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public boolean isParallel() {
    return parallel;
  }

  public ImgTileAssemblyLayer setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json,
                                              Map<CharSequence, byte[]> rs) {
    return new ImgTileAssemblyLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ImgTileAssemblyLayer[] addRefs(ImgTileAssemblyLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayer::addRef)
        .toArray((x) -> new ImgTileAssemblyLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileAssemblyLayer[][] addRefs(ImgTileAssemblyLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayer::addRefs)
        .toArray((x) -> new ImgTileAssemblyLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    if (1 == inObj.length) {
      return inObj[0];
    }
    int[] inputDimensions = inObj[0].getData().getDimensions();
    assert 3 == inputDimensions.length;
    final int length = inObj[0].getData().length();
    int[] outputDims = getOutputDims(inObj);
    final TensorList outputData = CudaSystem.run(gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert outputDims[0] > 0;
      assert outputDims[1] > 0;
      assert outputDims[2] > 0;
      @Nonnull final CudaMemory outputBuffer = gpu.allocate(
          (long) length * outputDims[2] * outputDims[1] * outputDims[0] * precision.size,
          MemoryType.Managed.ifEnabled(), false);
      int totalWidth = 0;
      int totalHeight = 0;
      int inputIndex = 0;
      RefList<CopyParams> copies = new RefArrayList<>();
      for (int row = 0; row < rows; row++) {
        int positionX = 0;
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          int[] tileDimensions = inObj[inputIndex].getData().getDimensions();
          rowHeight = Math.max(rowHeight, tileDimensions[1]);
          copies.add(new CopyParams(gpu, inObj, outputBuffer, length, outputDims, tileDimensions, inputIndex, positionX,
              totalHeight));
          positionX += tileDimensions[0];
          inputIndex += 1;
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        }
        totalHeight += rowHeight;
        totalWidth = Math.max(totalWidth, positionX);
      }
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      RefStream<CopyParams> stream = copies.stream();
      if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
        stream = stream.parallel();
      stream.forEach(this::copy);
      RefArrays.stream(inObj).forEach(r -> r.getData());
      CudaDevice.CudaTensorDescriptor descriptor = gpu.newTensorDescriptor(precision, length, outputDims[2],
          outputDims[1], outputDims[0]);
      CudaTensor ptr = new CudaTensor(outputBuffer, descriptor, precision);
      return new CudaTensorList(ptr, length, outputDims, precision);
    }, RefArrays.stream(inObj).map(Result::getData).toArray());

    return new Result(outputData, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
      if (!RefArrays.equals(error.getDimensions(), outputData.getDimensions())) {
        throw new AssertionError(RefArrays.toString(error.getDimensions()) + " != "
            + RefArrays.toString(outputData.getDimensions()));
      }
      if (error.length() != outputData.length()) {
        throw new AssertionError(error.length() + " != " + outputData.length());
      }
      assert error.length() == length;
      int totalHeight = 0;
      int inputIndex = 0;
      RefList<BackpropParams> tasks = new RefArrayList<>();
      for (int row = 0; row < rows; row++) {
        int positionX = 0;
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex];
          int[] tileDimensions = in.getData().getDimensions();
          rowHeight = Math.max(rowHeight, tileDimensions[1]);
          if (inObj[inputIndex].isAlive()) {
            tasks.add(new BackpropParams(inObj, buffer, error, outputDims, tileDimensions, length, positionX,
                totalHeight, inputIndex));
          }
          positionX += tileDimensions[0];
          inputIndex += 1;
        }
        totalHeight += rowHeight;
      }
      RefStream<BackpropParams> stream = tasks.stream();
      if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
        stream = stream.parallel();
      stream.forEach(this::backprop);
    }) {

      @Override
      public boolean isAlive() {
        return RefArrays.stream(inObj).anyMatch(x -> x.isAlive());
      }

      public void _free() {
      }
    };
  }

  public void backprop(final BackpropParams backpropParams) {
    final TensorList passbackTensorList = CudaSystem.run(gpu -> {
      CudaTensor ptr = copy(gpu, backpropParams.error, backpropParams.tileDimensions, backpropParams.outputDims,
          backpropParams.length, -backpropParams.positionX, -backpropParams.totalHeight);
      return new CudaTensorList(ptr, backpropParams.length, backpropParams.tileDimensions, precision);
    }, backpropParams.error);
    backpropParams.inObj[backpropParams.inputIndex].accumulate(backpropParams.buffer, passbackTensorList);
  }

  public CudaTensor copy(final CudnnHandle gpu, final TensorList error, final int[] tileDimensions,
                         final int[] outputDims, final int length, final int positionX, final int positionY) {
    @Nullable final CudaTensor errorPtr = gpu.getTensor(error, precision, MemoryType.Device, false);
    @Nonnull final CudaMemory passbackBuffer = gpu.allocate(
        (long) length * tileDimensions[2] * tileDimensions[1] * tileDimensions[0] * precision.size,
        MemoryType.Managed.ifEnabled(), false);
    copy(gpu, length, outputDims, errorPtr, tileDimensions, passbackBuffer, positionX, positionY);
    CudaDevice.CudaTensorDescriptor descriptor = gpu.newTensorDescriptor(precision, length, tileDimensions[2],
        tileDimensions[1], tileDimensions[0]);
    return new CudaTensor(passbackBuffer, descriptor, precision);
  }

  public void copy(final CopyParams copyParams) {
    CudnnHandle gpu = copyParams.gpu;
    gpu.initThread();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    @Nullable final CudaTensor inputBuffer = gpu.getTensor(copyParams.inObj[copyParams.inputIndex].getData(), precision,
        MemoryType.Device, false);
    copy(gpu, copyParams.length, copyParams.tileDimensions, inputBuffer, copyParams.outputDims, copyParams.outputBuffer,
        copyParams.positionX, copyParams.totalHeight);
  }

  public void copy(@Nonnull CudnnHandle gpu, int length, @Nonnull int[] sourceDimensions, @Nonnull CudaTensor source,
                   @Nonnull int[] destinationDimensions, @Nonnull CudaMemory destination, int positionX, int positionY) {
    if (3 != sourceDimensions.length)
      throw new IllegalArgumentException("inputDimensions.length");
    if (3 != destinationDimensions.length)
      throw new IllegalArgumentException("dimOut.length");
    int bands = sourceDimensions[2];
    if (bands != destinationDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, destinationDimensions[2]));
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(sourceDimensions, destinationDimensions,
        new int[]{positionX, positionY, 0});
    @Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        source.descriptor.nStride, //
        source.descriptor.cStride, //
        source.descriptor.hStride, //
        source.descriptor.wStride);
    @Nonnull final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        destinationDimensions[2] * destinationDimensions[1] * destinationDimensions[0], //
        destinationDimensions[1] * destinationDimensions[0], //
        destinationDimensions[0], //
        1);
    int sourceOffset = 0;
    int destinationOffset = 0;

    if (positionX > 0) {
      destinationOffset += Math.abs(positionX);
    } else {
      sourceOffset += source.descriptor.wStride * Math.abs(positionX);
    }
    if (positionY > 0) {
      destinationOffset += destinationDimensions[0] * Math.abs((positionY));
    } else {
      sourceOffset += source.descriptor.hStride * (Math.abs(positionY));
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= (source.descriptor.nStride * length);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(destinationDimensions);

    CudaMemory sourceMemory = source.getMemory(gpu);
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        sourceMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(1.0),
        destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(destinationOffset * precision.size)));
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    sourceMemory.dirty();
    destination.dirty();

  }

  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim,
        i -> Math.min(sourceDimensions[i] + offset[i], destinationDimensions[i]) - Math.max(offset[i], 0));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("rows", rows);
    json.addProperty("columns", columns);
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
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
  ImgTileAssemblyLayer addRef() {
    return (ImgTileAssemblyLayer) super.addRef();
  }

  private int[] getOutputDims(final Result[] inObj) {
    int bands = inObj[0].getData().getDimensions()[2];
    int totalWidth = 0;
    int totalHeight = 0;
    int inputIndex = 0;
    for (int row = 0; row < rows; row++) {
      int positionX = 0;
      int rowHeight = 0;
      for (int col = 0; col < columns; col++) {
        int[] dimensions = inObj[inputIndex].getData().getDimensions();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        positionX += dimensions[0];
        inputIndex += 1;
      }
      totalHeight += rowHeight;
      totalWidth = Math.max(totalWidth, positionX);
    }
    return new int[]{totalWidth, totalHeight, bands};
  }

  private static @RefAware
  class CopyParams extends ReferenceCountingBase {
    public final int length;
    public final int[] outputDims;
    public final CudnnHandle gpu;
    public final CudaMemory outputBuffer;
    public final int totalHeight;
    public final int inputIndex;
    public final int positionX;
    public final int[] tileDimensions;
    @Nonnull
    public final Result[] inObj;

    private CopyParams(final CudnnHandle gpu, @Nonnull final Result[] inObj, final CudaMemory outputBuffer,
                       final int length, final int[] outputDims, final int[] tileDimensions, final int inputIndex, final int positionX,
                       final int totalHeight) {
      this.length = length;
      this.outputDims = outputDims;
      this.gpu = gpu;
      this.outputBuffer = outputBuffer;
      this.totalHeight = totalHeight;
      this.inputIndex = inputIndex;
      this.positionX = positionX;
      this.tileDimensions = tileDimensions;
      this.inObj = inObj;
    }

    public static @SuppressWarnings("unused")
    CopyParams[] addRefs(CopyParams[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(CopyParams::addRef)
          .toArray((x) -> new CopyParams[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    CopyParams addRef() {
      return (CopyParams) super.addRef();
    }

  }

  private static @RefAware
  class BackpropParams extends ReferenceCountingBase {
    @Nonnull
    public final Result[] inObj;
    @Nonnull
    public final DeltaSet<UUID> buffer;
    @Nonnull
    public final TensorList error;
    public final int[] outputDims;
    public final int[] tileDimensions;
    public final int length;
    public final int positionX;
    public final int totalHeight;
    public final int inputIndex;

    private BackpropParams(@Nonnull final Result[] inObj, @Nonnull final DeltaSet<UUID> buffer,
                           @Nonnull final TensorList error, final int[] outputDims, final int[] tileDimensions, final int length,
                           final int positionX, final int totalHeight, final int inputIndex) {
      this.inObj = inObj;
      this.buffer = buffer;
      this.error = error;
      this.outputDims = outputDims;
      this.tileDimensions = tileDimensions;
      this.length = length;
      this.positionX = positionX;
      this.totalHeight = totalHeight;
      this.inputIndex = inputIndex;
    }

    public static @SuppressWarnings("unused")
    BackpropParams[] addRefs(BackpropParams[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(BackpropParams::addRef)
          .toArray((x) -> new BackpropParams[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    BackpropParams addRef() {
      return (BackpropParams) super.addRef();
    }

  }
}
