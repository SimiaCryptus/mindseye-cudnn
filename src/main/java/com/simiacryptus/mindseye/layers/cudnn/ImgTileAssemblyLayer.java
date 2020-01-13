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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgTileAssemblyLayer extends LayerBase implements MultiPrecision<ImgTileAssemblyLayer> {
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

  protected ImgTileAssemblyLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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
    return this.addRef();
  }

  public boolean isParallel() {
    return parallel;
  }

  public ImgTileAssemblyLayer setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileAssemblyLayer(json, rs);
  }

  public static @SuppressWarnings("unused") ImgTileAssemblyLayer[] addRefs(ImgTileAssemblyLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayer::addRef)
        .toArray((x) -> new ImgTileAssemblyLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgTileAssemblyLayer[][] addRefs(ImgTileAssemblyLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayer::addRefs)
        .toArray((x) -> new ImgTileAssemblyLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_09_0017 = getCompatibilityLayer();
      Result temp_09_0012 = temp_09_0017.eval(Result.addRefs(inObj));
      if (null != temp_09_0017)
        temp_09_0017.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_09_0012;
    }
    if (1 == inObj.length) {
      Result temp_09_0013 = inObj[0];
      ReferenceCounting.freeRefs(inObj);
      return temp_09_0013;
    }
    TensorList temp_09_0018 = inObj[0].getData();
    int[] inputDimensions = temp_09_0018.getDimensions();
    if (null != temp_09_0018)
      temp_09_0018.freeRef();
    assert 3 == inputDimensions.length;
    TensorList temp_09_0019 = inObj[0].getData();
    final int length = temp_09_0019.length();
    if (null != temp_09_0019)
      temp_09_0019.freeRef();
    int[] outputDims = getOutputDims(Result.addRefs(inObj));
    final ImgTileAssemblyLayer imgTileAssemblyLayer = this.addRef();
    final TensorList outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert outputDims[0] > 0;
      assert outputDims[1] > 0;
      assert outputDims[2] > 0;
      @Nonnull
      final CudaMemory outputBuffer = gpu.allocate(
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
          if (null != temp_09_0020)
            temp_09_0020.freeRef();
          rowHeight = Math.max(rowHeight, tileDimensions[1]);
          copies.add(new CopyParams(gpu, Result.addRefs(inObj), outputBuffer == null ? null : outputBuffer.addRef(),
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
      if (null != copies)
        copies.freeRef();
      if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
        stream = stream.parallel();
      stream.forEach(imgTileAssemblyLayer::copy);
      RefArrays.stream(Result.addRefs(inObj)).forEach(r -> {
        RefUtil.freeRef(r.getData());
        if (null != r)
          r.freeRef();
      });
      CudaDevice.CudaTensorDescriptor descriptor = gpu.newTensorDescriptor(precision, length, outputDims[2],
          outputDims[1], outputDims[0]);
      CudaTensor ptr = new CudaTensor(outputBuffer == null ? null : outputBuffer,
          descriptor == null ? null : descriptor.addRef(), precision);
      if (null != descriptor)
        descriptor.freeRef();
      CudaTensorList temp_09_0007 = new CudaTensorList(ptr == null ? null : ptr.addRef(), length, outputDims,
          precision);
      if (null != ptr)
        ptr.freeRef();
      return temp_09_0007;
    }, imgTileAssemblyLayer == null ? null : imgTileAssemblyLayer.addRef(), Result.addRefs(inObj)),
        RefArrays.stream(Result.addRefs(inObj)).map(Result::getData).toArray());

    try {
      try {
        try {
          return new Result(outputData, new Result.Accumulator() {
            {
              Result.addRefs(inObj);
            }

            @Override
            public void accept(DeltaSet<UUID> buffer, TensorList error) {
              if (!RefArrays.equals(error.getDimensions(), outputData.getDimensions())) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_09_0015 = new AssertionError(RefArrays.toString(error.getDimensions()) + " != "
                    + RefArrays.toString(outputData.getDimensions()));
                if (null != error)
                  error.freeRef();
                throw temp_09_0015;
              }
              if (error.length() != outputData.length()) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_09_0016 = new AssertionError(error.length() + " != " + outputData.length());
                if (null != error)
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
                  if (null != temp_09_0021)
                    temp_09_0021.freeRef();
                  if (null != in)
                    in.freeRef();
                  rowHeight = Math.max(rowHeight, tileDimensions[1]);
                  if (inObj[inputIndex].isAlive()) {
                    tasks.add(new BackpropParams(Result.addRefs(inObj), buffer == null ? null : buffer.addRef(),
                        error == null ? null : error.addRef(), outputDims, tileDimensions, length, positionX,
                        totalHeight, inputIndex));
                  }
                  positionX += tileDimensions[0];
                  inputIndex += 1;
                }
                totalHeight += rowHeight;
              }
              if (null != error)
                error.freeRef();
              if (null != buffer)
                buffer.freeRef();
              RefStream<BackpropParams> stream = tasks.stream();
              if (null != tasks)
                tasks.freeRef();
              if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
                stream = stream.parallel();
              stream.forEach(imgTileAssemblyLayer::backprop);
            }

            public @SuppressWarnings("unused") void _free() {
              ReferenceCounting.freeRefs(inObj);
            }
          }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                boolean temp_09_0009 = x.isAlive();
                if (null != x)
                  x.freeRef();
                return temp_09_0009;
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
      if (null != imgTileAssemblyLayer)
        imgTileAssemblyLayer.freeRef();
    }
  }

  public void backprop(final BackpropParams backpropParams) {
    final TensorList passbackTensorList = CudaSystem
        .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor ptr = copy(gpu, backpropParams.error.addRef(), backpropParams.tileDimensions,
              backpropParams.outputDims, backpropParams.length, -backpropParams.positionX, -backpropParams.totalHeight);
          CudaTensorList temp_09_0010 = new CudaTensorList(ptr == null ? null : ptr.addRef(), backpropParams.length,
              backpropParams.tileDimensions, precision);
          if (null != ptr)
            ptr.freeRef();
          return temp_09_0010;
        }, backpropParams == null ? null : backpropParams.addRef()), backpropParams.error.addRef());
    backpropParams.inObj[backpropParams.inputIndex].accumulate(backpropParams.buffer.addRef(),
        passbackTensorList == null ? null : passbackTensorList.addRef());
    if (null != backpropParams)
      backpropParams.freeRef();
    if (null != passbackTensorList)
      passbackTensorList.freeRef();
  }

  public CudaTensor copy(final CudnnHandle gpu, final TensorList error, final int[] tileDimensions,
      final int[] outputDims, final int length, final int positionX, final int positionY) {
    @Nullable
    final CudaTensor errorPtr = gpu.getTensor(error == null ? null : error.addRef(), precision, MemoryType.Device,
        false);
    if (null != error)
      error.freeRef();
    @Nonnull
    final CudaMemory passbackBuffer = gpu.allocate(
        (long) length * tileDimensions[2] * tileDimensions[1] * tileDimensions[0] * precision.size,
        MemoryType.Managed.ifEnabled(), false);
    copy(gpu, length, outputDims, errorPtr == null ? null : errorPtr.addRef(), tileDimensions,
        passbackBuffer == null ? null : passbackBuffer.addRef(), positionX, positionY);
    if (null != errorPtr)
      errorPtr.freeRef();
    CudaDevice.CudaTensorDescriptor descriptor = gpu.newTensorDescriptor(precision, length, tileDimensions[2],
        tileDimensions[1], tileDimensions[0]);
    CudaTensor temp_09_0011 = new CudaTensor(passbackBuffer == null ? null : passbackBuffer,
        descriptor == null ? null : descriptor.addRef(), precision);
    if (null != descriptor)
      descriptor.freeRef();
    return temp_09_0011;
  }

  public void copy(final CopyParams copyParams) {
    CudnnHandle gpu = copyParams.gpu;
    gpu.initThread();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    @Nullable
    final CudaTensor inputBuffer = gpu.getTensor(copyParams.inObj[copyParams.inputIndex].getData(), precision,
        MemoryType.Device, false);
    copy(gpu, copyParams.length, copyParams.tileDimensions, inputBuffer == null ? null : inputBuffer.addRef(),
        copyParams.outputDims, copyParams.outputBuffer.addRef(), copyParams.positionX, copyParams.totalHeight);
    if (null != copyParams)
      copyParams.freeRef();
    if (null != inputBuffer)
      inputBuffer.freeRef();
  }

  public void copy(@Nonnull CudnnHandle gpu, int length, @Nonnull int[] sourceDimensions, @Nonnull CudaTensor source,
      @Nonnull int[] destinationDimensions, @Nonnull CudaMemory destination, int positionX, int positionY) {
    if (3 != sourceDimensions.length) {
      source.freeRef();
      destination.freeRef();
      throw new IllegalArgumentException("inputDimensions.length");
    }
    if (3 != destinationDimensions.length) {
      source.freeRef();
      destination.freeRef();
      throw new IllegalArgumentException("dimOut.length");
    }
    int bands = sourceDimensions[2];
    if (bands != destinationDimensions[2]) {
      source.freeRef();
      destination.freeRef();
      throw new IllegalArgumentException(RefString.format("%d != %d", bands, destinationDimensions[2]));
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull
    final int[] viewDim = getViewDimensions(sourceDimensions, destinationDimensions,
        new int[] { positionX, positionY, 0 });
    @Nonnull
    final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(precision, //
        length, //
        viewDim[2], //
        viewDim[1], //
        viewDim[0], //
        source.descriptor.nStride, //
        source.descriptor.cStride, //
        source.descriptor.hStride, //
        source.descriptor.wStride);
    @Nonnull
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
      destinationOffset += destinationDimensions[0] * Math.abs((positionY));
    } else {
      sourceOffset += source.descriptor.hStride * (Math.abs(positionY));
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= (source.descriptor.nStride * length);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(destinationDimensions);

    CudaMemory sourceMemory = source.getMemory(gpu);
    source.freeRef();
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), sourceViewDescriptor.getPtr(),
        sourceMemory.getPtr().withByteOffset(sourceOffset * precision.size), precision.getPointer(1.0),
        destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(destinationOffset * precision.size)));
    destinationViewDescriptor.freeRef();
    sourceViewDescriptor.freeRef();
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    RefUtil.freeRef(sourceMemory.dirty());
    if (null != sourceMemory)
      sourceMemory.freeRef();
    RefUtil.freeRef(destination.dirty());
    destination.freeRef();

  }

  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull
    final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim,
        i -> Math.min(sourceDimensions[i] + offset[i], destinationDimensions[i]) - Math.max(offset[i], 0));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgTileAssemblyLayer addRef() {
    return (ImgTileAssemblyLayer) super.addRef();
  }

  private int[] getOutputDims(final Result[] inObj) {
    TensorList temp_09_0022 = inObj[0].getData();
    int bands = temp_09_0022.getDimensions()[2];
    if (null != temp_09_0022)
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
        if (null != temp_09_0023)
          temp_09_0023.freeRef();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        positionX += dimensions[0];
        inputIndex += 1;
      }
      totalHeight += rowHeight;
      totalWidth = Math.max(totalWidth, positionX);
    }
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    return new int[] { totalWidth, totalHeight, bands };
  }

  private static class CopyParams extends ReferenceCountingBase {
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
      CudaMemory temp_09_0001 = outputBuffer == null ? null : outputBuffer.addRef();
      this.outputBuffer = temp_09_0001 == null ? null : temp_09_0001.addRef();
      if (null != temp_09_0001)
        temp_09_0001.freeRef();
      if (null != outputBuffer)
        outputBuffer.freeRef();
      this.totalHeight = totalHeight;
      this.inputIndex = inputIndex;
      this.positionX = positionX;
      this.tileDimensions = tileDimensions;
      Result[] temp_09_0002 = Result.addRefs(inObj);
      this.inObj = Result.addRefs(temp_09_0002);
      if (null != temp_09_0002)
        ReferenceCounting.freeRefs(temp_09_0002);
      ReferenceCounting.freeRefs(inObj);
    }

    public static @SuppressWarnings("unused") CopyParams[] addRefs(CopyParams[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(CopyParams::addRef).toArray((x) -> new CopyParams[x]);
    }

    public @SuppressWarnings("unused") void _free() {
      ReferenceCounting.freeRefs(inObj);
      if (null != outputBuffer)
        outputBuffer.freeRef();
    }

    public @Override @SuppressWarnings("unused") CopyParams addRef() {
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
      Result[] temp_09_0003 = Result.addRefs(inObj);
      this.inObj = Result.addRefs(temp_09_0003);
      if (null != temp_09_0003)
        ReferenceCounting.freeRefs(temp_09_0003);
      ReferenceCounting.freeRefs(inObj);
      DeltaSet<UUID> temp_09_0004 = buffer == null ? null : buffer.addRef();
      this.buffer = temp_09_0004 == null ? null : temp_09_0004.addRef();
      if (null != temp_09_0004)
        temp_09_0004.freeRef();
      buffer.freeRef();
      TensorList temp_09_0005 = error == null ? null : error.addRef();
      this.error = temp_09_0005 == null ? null : temp_09_0005.addRef();
      if (null != temp_09_0005)
        temp_09_0005.freeRef();
      error.freeRef();
      this.outputDims = outputDims;
      this.tileDimensions = tileDimensions;
      this.length = length;
      this.positionX = positionX;
      this.totalHeight = totalHeight;
      this.inputIndex = inputIndex;
    }

    public static @SuppressWarnings("unused") BackpropParams[] addRefs(BackpropParams[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(BackpropParams::addRef)
          .toArray((x) -> new BackpropParams[x]);
    }

    public @SuppressWarnings("unused") void _free() {
      error.freeRef();
      buffer.freeRef();
      ReferenceCounting.freeRefs(inObj);
    }

    public @Override @SuppressWarnings("unused") BackpropParams addRef() {
      return (BackpropParams) super.addRef();
    }

  }
}
