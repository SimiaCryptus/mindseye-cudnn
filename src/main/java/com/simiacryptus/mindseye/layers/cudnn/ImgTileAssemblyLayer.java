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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ImgTileAssemblyLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgTileAssemblyLayer.class);

  private int columns;
  private int rows;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private boolean parallel;

  private ImgTileAssemblyLayer() {
  }

  public ImgTileAssemblyLayer(int columns, int rows) {
    this.columns = columns;
    this.rows = rows;
  }

  protected ImgTileAssemblyLayer(@Nonnull final JsonObject json) {
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

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public boolean isParallel() {
    return parallel;
  }

  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileAssemblyLayer(json);
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
    if (1 == inObj.length) {
      Result result = inObj[0].addRef();
      RefUtil.freeRef(inObj);
      return result;
    }
    TensorList data0 = inObj[0].getData();
    int[] inputDimensions = data0.getDimensions();
    final int length = data0.length();
    data0.freeRef();
    assert 3 == inputDimensions.length;
    int[] outputDims = getOutputDims(RefUtil.addRef(inObj));
    final TensorList outputData = fwd(length, outputDims, RefUtil.addRef(inObj));
    Result.Accumulator accumulator = new Accumulator(this.addRef(), outputData.addRef(), length, outputDims, rows, columns, parallel, RefUtil.addRef(inObj));
    boolean isAlive = RefArrays.stream(inObj).anyMatch(x -> {
      boolean temp_09_0009 = x.isAlive();
      x.freeRef();
      return temp_09_0009;
    });
    return new Result(outputData, accumulator, isAlive);
  }

  public void backprop(@Nonnull final BackpropParams backpropParams) {
    final TensorList passbackTensorList = CudaSystem
        .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor ptr = copy(gpu, backpropParams.error.addRef(), backpropParams.tileDimensions,
              backpropParams.outputDims, backpropParams.length, -backpropParams.positionX, -backpropParams.totalHeight);
          CudaTensorList temp_09_0010 = new CudaTensorList(ptr == null ? null : ptr.addRef(), backpropParams.length,
              backpropParams.tileDimensions, precision);
          if (null != ptr)
            ptr.freeRef();
          return temp_09_0010;
        }, backpropParams.addRef()), backpropParams.error.addRef());
    backpropParams.inObj[backpropParams.inputIndex].accumulate(backpropParams.buffer.addRef(),
        passbackTensorList == null ? null : passbackTensorList.addRef());
    backpropParams.freeRef();
    if (null != passbackTensorList)
      passbackTensorList.freeRef();
  }

  @Nullable
  public CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nullable final TensorList error, @Nonnull final int[] tileDimensions,
                         @Nonnull final int[] outputDims, final int length, final int positionX, final int positionY) {
    @Nullable final CudaTensor errorPtr = gpu.getTensor(error == null ? null : error.addRef(), precision, MemoryType.Device,
        false);
    if (null != error)
      error.freeRef();
    @Nonnull final CudaMemory passbackBuffer = gpu.allocate(
        (long) length * tileDimensions[2] * tileDimensions[1] * tileDimensions[0] * precision.size,
        MemoryType.Managed.ifEnabled(), false);
    copy(gpu.addRef(), length, outputDims, errorPtr, tileDimensions,
        passbackBuffer.addRef(), positionX, positionY);
    CudaDevice.CudaTensorDescriptor descriptor = gpu.newTensorDescriptor(precision, length, tileDimensions[2],
        tileDimensions[1], tileDimensions[0]);
    gpu.freeRef();
    return new CudaTensor(passbackBuffer, descriptor, precision);
  }

  public void copy(@Nonnull final CopyParams copyParams) {
    CudnnHandle gpu = copyParams.gpu.addRef();
    gpu.initThread();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    @Nullable final CudaTensor inputBuffer = gpu.getTensor(copyParams.inObj[copyParams.inputIndex].getData(), precision,
        MemoryType.Device, false);
    copy(gpu, copyParams.length, copyParams.tileDimensions, inputBuffer,
        copyParams.outputDims, copyParams.outputBuffer.addRef(), copyParams.positionX, copyParams.totalHeight);
    copyParams.freeRef();
  }

  public void copy(@Nonnull CudnnHandle gpu, int length, @Nonnull int[] sourceDimensions, @Nonnull CudaTensor source,
                   @Nonnull int[] destinationDimensions, @Nonnull CudaMemory destination, int positionX, int positionY) {
    if (3 != sourceDimensions.length) {
      source.freeRef();
      destination.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("inputDimensions.length");
    }
    if (3 != destinationDimensions.length) {
      source.freeRef();
      destination.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    int bands = sourceDimensions[2];
    if (bands != destinationDimensions[2]) {
      source.freeRef();
      destination.freeRef();
      gpu.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", bands, destinationDimensions[2]));
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(sourceDimensions, destinationDimensions,
        new int[]{positionX, positionY, 0});
    final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        source.descriptor.nStride, //
        source.descriptor.cStride, //
        source.descriptor.hStride, //
        source.descriptor.wStride);
    final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(precision, //
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
      destinationOffset += destinationDimensions[0] * Math.abs(positionY);
    } else {
      sourceOffset += source.descriptor.hStride * Math.abs(positionY);
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= source.descriptor.nStride * length;
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(destinationDimensions);

    CudaMemory sourceMemory = source.getMemory(gpu.addRef());
    source.freeRef();
    assert sourceMemory != null;
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        sourceMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(1.0),
        destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(destinationOffset * precision.size)));
    destinationViewDescriptor.freeRef();
    sourceViewDescriptor.freeRef();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    sourceMemory.dirty();
    sourceMemory.freeRef();
    gpu.freeRef();
    destination.dirty();
    destination.freeRef();
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
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileAssemblyLayer addRef() {
    return (ImgTileAssemblyLayer) super.addRef();
  }

  @NotNull
  private TensorList fwd(int length, int[] outputDims, @Nonnull Result[] inObj) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
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
              TensorList temp_09_0020 = inObj[inputIndex].getData();
              int[] tileDimensions = temp_09_0020.getDimensions();
              temp_09_0020.freeRef();
              rowHeight = Math.max(rowHeight, tileDimensions[1]);
              copies.add(new CopyParams(gpu.addRef(), RefUtil.addRef(inObj), outputBuffer.addRef(),
                  length, outputDims, tileDimensions, inputIndex, positionX, totalHeight));
              positionX += tileDimensions[0];
              inputIndex += 1;
              assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            }
            totalHeight += rowHeight;
            totalWidth = Math.max(totalWidth, positionX);
          }
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          RefStream<CopyParams> stream = copies.stream();
          if (!CoreSettings.INSTANCE().singleThreaded && parallel)
            stream = stream.parallel();
          stream.forEach(copyParams -> copy(copyParams));
          copies.freeRef();
          CudaTensorList cudaTensorList = new CudaTensorList(
              new CudaTensor(outputBuffer,
                  gpu.newTensorDescriptor(precision, length, outputDims[2], outputDims[1], outputDims[0]),
                  precision),
              length, outputDims, precision);
          gpu.freeRef();
          return cudaTensorList;
        }, RefUtil.addRef(inObj)),
        RefArrays.stream(inObj).map(result -> {
          TensorList data = result.getData();
          result.freeRef();
          return data;
        }).toArray());
  }

  @Nonnull
  private int[] getOutputDims(@Nullable final Result[] inObj) {
    assert inObj != null;
    TensorList temp_09_0022 = inObj[0].getData();
    int bands = temp_09_0022.getDimensions()[2];
    temp_09_0022.freeRef();
    int totalWidth = 0;
    int totalHeight = 0;
    int inputIndex = 0;
    for (int row = 0; row < rows; row++) {
      int positionX = 0;
      int rowHeight = 0;
      for (int col = 0; col < columns; col++) {
        TensorList temp_09_0023 = inObj[inputIndex].getData();
        int[] dimensions = temp_09_0023.getDimensions();
        temp_09_0023.freeRef();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        positionX += dimensions[0];
        inputIndex += 1;
      }
      totalHeight += rowHeight;
      totalWidth = Math.max(totalWidth, positionX);
    }
    RefUtil.freeRef(inObj);
    return new int[]{totalWidth, totalHeight, bands};
  }

  private static class CopyParams extends ReferenceCountingBase {
    public final int length;
    public final int[] outputDims;
    public final CudnnHandle gpu;
    @Nonnull
    public final CudaMemory outputBuffer;
    public final int totalHeight;
    public final int inputIndex;
    public final int positionX;
    public final int[] tileDimensions;
    @Nonnull
    public final Result[] inObj;

    private CopyParams(final CudnnHandle gpu, @Nonnull final Result[] inObj, @Nullable final CudaMemory outputBuffer,
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

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
      if (null != gpu) gpu.freeRef();
      outputBuffer.freeRef();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    CopyParams addRef() {
      return (CopyParams) super.addRef();
    }
  }

  private static class BackpropParams extends ReferenceCountingBase {
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

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      error.freeRef();
      buffer.freeRef();
      RefUtil.freeRef(inObj);
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    BackpropParams addRef() {
      return (BackpropParams) super.addRef();
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final ImgTileAssemblyLayer imgTileAssemblyLayer;
    private final TensorList outputData;
    private final int length;
    private final int[] outputDims;
    private final Result[] inObj;
    private int rows;
    private int columns;
    private boolean parallel;

    public Accumulator(ImgTileAssemblyLayer imgTileAssemblyLayer, TensorList outputData, int length, int[] outputDims, int rows, int columns, boolean parallel, Result... inObj) {
      this.imgTileAssemblyLayer = imgTileAssemblyLayer;
      this.outputData = outputData;
      this.length = length;
      this.outputDims = outputDims;
      this.inObj = inObj;
      this.rows = rows;
      this.columns = columns;
      this.parallel = parallel;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      if (!RefArrays.equals(error.getDimensions(), outputData.getDimensions())) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_09_0015 = new AssertionError(RefArrays.toString(error.getDimensions()) + " != "
            + RefArrays.toString(outputData.getDimensions()));
        error.freeRef();
        throw temp_09_0015;
      }
      if (error.length() != outputData.length()) {
        if (null != buffer)
          buffer.freeRef();
        AssertionError temp_09_0016 = new AssertionError(error.length() + " != " + outputData.length());
        error.freeRef();
        throw temp_09_0016;
      }
      assert error.length() == length;
      int totalHeight = 0;
      int inputIndex = 0;
      RefList<BackpropParams> tasks = new RefArrayList<>();
      for (int row = 0; row < rows; row++) {
        int positionX = 0;
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex].addRef();
          TensorList temp_09_0021 = in.getData();
          int[] tileDimensions = temp_09_0021.getDimensions();
          temp_09_0021.freeRef();
          in.freeRef();
          rowHeight = Math.max(rowHeight, tileDimensions[1]);
          if (inObj[inputIndex].isAlive()) {
            tasks.add(new BackpropParams(RefUtil.addRef(inObj), buffer == null ? null : buffer.addRef(),
                error.addRef(), outputDims, tileDimensions, length, positionX,
                totalHeight, inputIndex));
          }
          positionX += tileDimensions[0];
          inputIndex += 1;
        }
        totalHeight += rowHeight;
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
      RefStream<BackpropParams> stream = tasks.stream();
      tasks.freeRef();
      if (!CoreSettings.INSTANCE().singleThreaded && parallel)
        stream = stream.parallel();
      stream.forEach(backpropParams -> imgTileAssemblyLayer.backprop(backpropParams));
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
      imgTileAssemblyLayer.freeRef();
      outputData.freeRef();
    }
  }
}
