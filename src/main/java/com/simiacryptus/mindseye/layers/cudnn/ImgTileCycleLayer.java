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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgTileCycleLayer extends LayerBase implements MultiPrecision<ImgTileCycleLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileCycleLayer.class);
  private double xPos = 0.5;
  private double yPos = 0.5;

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public ImgTileCycleLayer() {
  }

  protected ImgTileCycleLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgTileCycleLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public double getxPos() {
    return xPos;
  }

  public double getyPos() {
    return yPos;
  }

  @Nonnull
  public ImgTileCycleLayer setXPos(double xPos) {
    this.xPos = xPos;
    return this.addRef();
  }

  @Nonnull
  public ImgTileCycleLayer setYPos(double yPos) {
    this.yPos = yPos;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileCycleLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileCycleLayer(json, rs);
  }

  @Nonnull
  public static CudaTensor copy(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor input, final int length,
                                @Nonnull Precision precision, final int splitX, final int splitY) {
    CudaMemory inputTensorMemory = input.getMemory(gpu.addRef());
    @Nonnull final CudaDevice.CudaTensorDescriptor imageDescriptor = gpu.newTensorDescriptor(precision, length,
        input.descriptor.channels, input.descriptor.height, input.descriptor.width, input.descriptor.nStride,
        input.descriptor.cStride, input.descriptor.hStride, input.descriptor.wStride);
    @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) length * imageDescriptor.nStride * precision.size,
        MemoryType.Managed.ifEnabled(), true);
    int splitY2 = input.descriptor.height - splitY;
    int splitX2 = input.descriptor.width - splitX;
    {
      @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(precision, length,
          input.descriptor.channels, splitY, splitX, input.descriptor.nStride, input.descriptor.cStride,
          input.descriptor.hStride, input.descriptor.wStride);
      assert inputTensorMemory != null;
      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), tileDescriptor.getPtr(),
          inputTensorMemory.getPtr().withByteOffset(0 * precision.size), precision.getPointer(0.0),
          tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(
              (splitY2 * input.descriptor.hStride + splitX2 * input.descriptor.wStride) * precision.size)));
      tileDescriptor.freeRef();
    }
    {
      @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(precision, length,
          input.descriptor.channels, splitY2, splitX, input.descriptor.nStride, input.descriptor.cStride,
          input.descriptor.hStride, input.descriptor.wStride);
      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), tileDescriptor.getPtr(),
          inputTensorMemory.getPtr().withByteOffset(splitY * input.descriptor.hStride * precision.size),
          precision.getPointer(0.0), tileDescriptor.getPtr(),
          outputBuffer.getPtr().withByteOffset(splitX2 * input.descriptor.wStride * precision.size)));
      tileDescriptor.freeRef();
    }
    {
      @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(precision, length,
          input.descriptor.channels, splitY, splitX2, input.descriptor.nStride, input.descriptor.cStride,
          input.descriptor.hStride, input.descriptor.wStride);
      CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), tileDescriptor.getPtr(),
          inputTensorMemory.getPtr().withByteOffset(splitX * input.descriptor.wStride * precision.size),
          precision.getPointer(0.0), tileDescriptor.getPtr(),
          outputBuffer.getPtr().withByteOffset(splitY2 * input.descriptor.hStride * precision.size)));
      tileDescriptor.freeRef();
    }
    @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(precision, length,
        input.descriptor.channels, splitY2, splitX2, input.descriptor.nStride, input.descriptor.cStride,
        input.descriptor.hStride, input.descriptor.wStride);
    CudaSystem.handle(gpu.cudnnTransformTensor(precision.getPointer(1.0), tileDescriptor.getPtr(),
        inputTensorMemory.getPtr()
            .withByteOffset((splitY * input.descriptor.hStride + splitX * input.descriptor.wStride) * precision.size),
        precision.getPointer(0.0), tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(0 * precision.size)));
    gpu.freeRef();
    tileDescriptor.freeRef();
    RefUtil.freeRef(inputTensorMemory.dirty());
    RefUtil.freeRef(outputBuffer.dirty());
    inputTensorMemory.freeRef();
    CudaTensor temp_49_0001 = new CudaTensor(outputBuffer, imageDescriptor, precision);
    input.freeRef();
    return temp_49_0001;
  }

  @Nonnull
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim, i -> Math.min(sourceDimensions[i], destinationDimensions[i]));
    return viewDim;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileCycleLayer[] addRefs(@Nullable ImgTileCycleLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayer::addRef)
        .toArray((x) -> new ImgTileCycleLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileCycleLayer[][] addRefs(@Nullable ImgTileCycleLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayer::addRefs)
        .toArray((x) -> new ImgTileCycleLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_49_0011 = getCompatibilityLayer();
      Result temp_49_0007 = temp_49_0011.eval(Result.addRefs(inObj));
      temp_49_0011.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_49_0007;
    }
    assert 1 == inObj.length;
    final Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull
    int[] dimIn = inputData.getDimensions();
    int splitX1 = (int) (dimIn[0] * getxPos());
    int splitX2 = dimIn[0] - splitX1;
    int splitY1 = (int) (dimIn[1] * getyPos());
    int splitY2 = dimIn[1] - splitY1;
    final TensorList outputData = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, false);
      CudaTensor cudaTensor = copy(gpu, inputTensor.addRef(), length, precision, splitX1, splitY1);
      inputTensor.freeRef();
      CudaTensorList temp_49_0003 = new CudaTensorList(cudaTensor.addRef(), length, dimIn, precision);
      cudaTensor.freeRef();
      return temp_49_0003;
    }, inputData.addRef()), inputData.addRef());
    inputData.freeRef();
    try {
      try {
        try {
          return new Result(outputData, new Result.Accumulator() {
            {
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
              if (!RefArrays.equals(delta.getDimensions(), outputData.getDimensions())) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_49_0009 = new AssertionError(RefArrays.toString(delta.getDimensions()) + " != "
                    + RefArrays.toString(outputData.getDimensions()));
                delta.freeRef();
                throw temp_49_0009;
              }
              if (delta.length() != outputData.length()) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_49_0010 = new AssertionError(delta.length() + " != " + outputData.length());
                delta.freeRef();
                throw temp_49_0010;
              }
              assert delta.length() == length;
              if (input.isAlive()) {
                final TensorList passbackTensorList = CudaSystem
                    .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                      @Nullable final CudaTensor errorPtr = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, false);
                      CudaTensor cudaTensor = copy(gpu, errorPtr.addRef(), length, precision, splitX2, splitY2);
                      errorPtr.freeRef();
                      CudaTensorList temp_49_0005 = new CudaTensorList(cudaTensor.addRef(), length, dimIn, precision);
                      cudaTensor.freeRef();
                      return temp_49_0005;
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
          }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                boolean temp_49_0006 = x.isAlive();
                x.freeRef();
                return temp_49_0006;
              });
            }

            @Override
            public void accumulate(@Nullable final DeltaSet<UUID> buffer, @Nullable final TensorList delta) {
              Result.Accumulator temp_49_0012 = getAccumulator();
              assert temp_49_0012 != null;
              temp_49_0012.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
              temp_49_0012.freeRef();
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
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
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  ImgTileCycleLayer addRef() {
    return (ImgTileCycleLayer) super.addRef();
  }
}
