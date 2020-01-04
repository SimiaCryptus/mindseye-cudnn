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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgTileSubnetLayer extends WrapperLayer
    implements MultiPrecision<ImgTileSubnetLayer> {

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
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
  }

  protected ImgTileSubnetLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
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
    return this;
  }

  public boolean isParallel() {
    return parallel;
  }

  public ImgTileSubnetLayer setParallel(boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0];
    TensorList inputData = input.getData();
    @Nonnull
    final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int bands = inputDims[2];
    int length = inputData.length();
    CudaTensor passback = CudaSystem.run(gpu -> {
      return new CudaTensor(
          gpu.allocate(inputData.getElements() * precision.size, MemoryType.Managed.ifEnabled(), true),
          gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
    });
    {
      int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
      int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
      if (cols == 1 && rows == 1)
        return getInner().eval(inObj);
      int[] tileDimensions = { width, height, bands };
      AtomicInteger counter = new AtomicInteger(0);
      Result[][] tileResults = new Result[rows][];
      for (int row = 0; row < rows; row++) {
        tileResults[row] = new Result[cols];
        for (int col = 0; col < cols; col++) {
          int positionX = col * strideX;
          int positionY = row * strideY;
          assert positionX >= 0;
          assert positionY >= 0;
          assert positionX < inputDims[0];
          assert positionY < inputDims[1];

          CudaTensor tile = CudaSystem.run(gpu -> {
            return ImgTileSelectLayer.copy(gpu, inputData, inputData.getDimensions(), tileDimensions, precision,
                positionX, positionY, true);
          });

          tileResults[row][col] = getInner().eval(new Result(
              new CudaTensorList(tile, length, tileDimensions, precision), (DeltaSet<UUID> ctx, TensorList delta) -> {
                CudaSystem.run(gpu -> {
                  ImgTileSelectLayer.copy(gpu, delta, tileDimensions, -positionX, -positionY, precision, passback);
                });
                if (counter.incrementAndGet() >= rows * cols) {
                  counter.set(0);
                  input.accumulate(ctx, new CudaTensorList(passback, length, inputDims, precision));
                }
              }) {
            public void _free() {
              super._free();
            }
          });
        }
      }
      logger.debug(String.format("Broke input %s into %s rows, %s cols",
          com.simiacryptus.ref.wrappers.RefArrays.toString(inputDims), rows, cols));
      Result result = new ImgTileAssemblyLayer(cols, rows).setParallel(parallel).setPrecision(precision)
          .eval(com.simiacryptus.ref.wrappers.RefArrays.stream(tileResults)
              .flatMap(com.simiacryptus.ref.wrappers.RefArrays::stream).<Result>toArray(i -> new Result[i]));
      return new Result(result.getData(), (ctx, delta) -> {
        result.accumulate(ctx, delta);
      }) {

        @Override
        public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
          getAccumulator().accept(buffer, delta);
        }

        public void _free() {
          super._free();
        }
      };
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJson(resources, dataSerializer);
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
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    getInner().setFrozen(frozen);
    return super.setFrozen(frozen);
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") ImgTileSubnetLayer addRef() {
    return (ImgTileSubnetLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgTileSubnetLayer[] addRefs(ImgTileSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRef)
        .toArray((x) -> new ImgTileSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgTileSubnetLayer[][] addRefs(ImgTileSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRefs)
        .toArray((x) -> new ImgTileSubnetLayer[x][]);
  }
}
