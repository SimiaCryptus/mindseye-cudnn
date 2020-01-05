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
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefConsumer;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

@SuppressWarnings("serial")
public @RefAware
class ImgTileSubnetLayer extends WrapperLayer implements MultiPrecision<ImgTileSubnetLayer> {

  private static final Logger logger = LoggerFactory.getLogger(ImgTileSubnetLayer.class);
  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX,
                            final int strideY) {
    super(subnetwork);
    if (null != subnetwork)
      subnetwork.freeRef();
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
    if (null != subnetwork)
      subnetwork.freeRef();
  }

  protected ImgTileSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    height = json.getAsJsonPrimitive("height").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    this.parallel = json.get("parallel").getAsBoolean();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgTileSubnetLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public boolean isParallel() {
    return parallel;
  }

  public ImgTileSubnetLayer setParallel(boolean parallel) {
    this.parallel = parallel;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ImgTileSubnetLayer[] addRefs(ImgTileSubnetLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRef)
        .toArray((x) -> new ImgTileSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileSubnetLayer[][] addRefs(ImgTileSubnetLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRefs)
        .toArray((x) -> new ImgTileSubnetLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0].addRef();
    TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int bands = inputDims[2];
    int length = inputData.length();
    CudaTensor passback = CudaSystem.run(RefUtil.wrapInterface(
        (Function<CudnnHandle, CudaTensor>) gpu -> {
          return new CudaTensor(
              gpu.allocate(inputData.getElements() * precision.size, MemoryType.Managed.ifEnabled(), true),
              gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
        }, inputData == null ? null : inputData.addRef()));
    {
      int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
      int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
      if (cols == 1 && rows == 1) {
        if (null != input)
          input.freeRef();
        if (null != inputData)
          inputData.freeRef();
        if (null != passback)
          passback.freeRef();
        Layer temp_03_0006 = getInner();
        Result temp_03_0004 = temp_03_0006
            .eval(Result.addRefs(inObj));
        if (null != temp_03_0006)
          temp_03_0006.freeRef();
        ReferenceCounting.freeRefs(inObj);
        return temp_03_0004;
      }
      int[] tileDimensions = {width, height, bands};
      AtomicInteger counter = new AtomicInteger(0);
      Result[][] tileResults = new Result[rows][];
      for (int row = 0; row < rows; row++) {
        {
          Result[] temp_03_0001 = new Result[cols];
          if (null != tileResults[row])
            ReferenceCounting.freeRefs(tileResults[row]);
          tileResults[row] = Result.addRefs(temp_03_0001);
          if (null != temp_03_0001)
            ReferenceCounting.freeRefs(temp_03_0001);
        }
        for (int col = 0; col < cols; col++) {
          int positionX = col * strideX;
          int positionY = row * strideY;
          assert positionX >= 0;
          assert positionY >= 0;
          assert positionX < inputDims[0];
          assert positionY < inputDims[1];

          CudaTensor tile = CudaSystem.run(RefUtil.wrapInterface(
              (Function<CudnnHandle, CudaTensor>) gpu -> {
                return ImgTileSelectLayer.copy(gpu, inputData == null ? null : inputData.addRef(),
                    inputData.getDimensions(), tileDimensions, precision, positionX, positionY, true);
              }, inputData == null ? null : inputData.addRef()));

          {
            Layer temp_03_0007 = getInner();
            Result temp_03_0002 = temp_03_0007.eval(
                new Result(new CudaTensorList(tile == null ? null : tile.addRef(), length, tileDimensions, precision),
                    new Result.Accumulator() {
                      {
                      }

                      @Override
                      public void accept(DeltaSet<UUID> ctx, TensorList delta) {
                        CudaSystem.run(RefUtil.wrapInterface(
                            (RefConsumer<CudnnHandle>) gpu -> {
                              ImgTileSelectLayer.copy(gpu, delta == null ? null : delta.addRef(), tileDimensions,
                                  -positionX, -positionY, precision, passback == null ? null : passback.addRef());
                            }, delta == null ? null : delta.addRef(), passback == null ? null : passback.addRef()));
                        if (null != delta)
                          delta.freeRef();
                        if (counter.incrementAndGet() >= rows * cols) {
                          counter.set(0);
                          input.accumulate(ctx == null ? null : ctx.addRef(), new CudaTensorList(
                              passback == null ? null : passback.addRef(), length, inputDims, precision));
                        }
                        if (null != ctx)
                          ctx.freeRef();
                      }

                      public @SuppressWarnings("unused")
                      void _free() {
                      }
                    }) {
                  public void _free() {
                    super._free();
                  }
                });
            if (null != temp_03_0007)
              temp_03_0007.freeRef();
            if (null != tileResults[row][col])
              tileResults[row][col].freeRef();
            tileResults[row][col] = temp_03_0002 == null ? null : temp_03_0002.addRef();
            if (null != temp_03_0002)
              temp_03_0002.freeRef();
          }
          if (null != tile)
            tile.freeRef();
        }
      }
      logger.debug(String.format("Broke input %s into %s rows, %s cols", RefArrays.toString(inputDims), rows, cols));
      ImgTileAssemblyLayer temp_03_0005 = new ImgTileAssemblyLayer(cols, rows);
      ImgTileAssemblyLayer temp_03_0008 = temp_03_0005.setParallel(parallel);
      ImgTileAssemblyLayer temp_03_0009 = temp_03_0008.setPrecision(precision);
      Result result = temp_03_0009.eval(RefArrays.stream(Result.addRefs(tileResults))
          .flatMap(RefArrays::stream).<Result>toArray(i -> new Result[i]));
      if (null != temp_03_0009)
        temp_03_0009.freeRef();
      if (null != temp_03_0008)
        temp_03_0008.freeRef();
      if (null != temp_03_0005)
        temp_03_0005.freeRef();
      if (null != tileResults)
        ReferenceCounting.freeRefs(tileResults);
      if (null != input)
        input.freeRef();
      if (null != inputData)
        inputData.freeRef();
      if (null != passback)
        passback.freeRef();
      try {
        ReferenceCounting.freeRefs(inObj);
        return new Result(result.getData(), new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> ctx, TensorList delta) {
            result.accumulate(ctx == null ? null : ctx.addRef(), delta == null ? null : delta.addRef());
            if (null != delta)
              delta.freeRef();
            if (null != ctx)
              ctx.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }) {

          @Override
          public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
            Result.Accumulator temp_03_0010 = getAccumulator();
            temp_03_0010.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
            if (null != temp_03_0010)
              temp_03_0010.freeRef();
            if (null != delta)
              delta.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public void _free() {
            super._free();
          }
        };
      } finally {
        if (null != result)
          result.freeRef();
      }
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("height", height);
    json.addProperty("width", width);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    Layer temp_03_0011 = getInner();
    RefUtil.freeRef(temp_03_0011.setFrozen(frozen));
    if (null != temp_03_0011)
      temp_03_0011.freeRef();
    return super.setFrozen(frozen);
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  ImgTileSubnetLayer addRef() {
    return (ImgTileSubnetLayer) super.addRef();
  }
}
