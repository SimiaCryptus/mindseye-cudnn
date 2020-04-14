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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Img tile subnet layer.
 */
@SuppressWarnings("serial")
public class ImgTileSubnetLayer extends WrapperLayer implements MultiPrecision {

  private static final Logger logger = LoggerFactory.getLogger(ImgTileSubnetLayer.class);
  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private boolean parallel = true;

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the height
   * @param strideX    the stride x
   * @param strideY    the stride y
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX,
                            final int strideY) {
    super(subnetwork);
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the height
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
  }

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   */
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

  @Override
  public void setPrecision(Precision precision) {
    this.precision = precision;
  }

  /**
   * Is parallel boolean.
   *
   * @return the boolean
   */
  public boolean isParallel() {
    return parallel;
  }

  /**
   * Sets parallel.
   *
   * @param parallel the parallel
   */
  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @Override
  public void setFrozen(final boolean frozen) {
    Layer inner = getInner();
    assert inner != null;
    inner.setFrozen(frozen);
    inner.freeRef();
  }

  /**
   * From json img tile subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img tile subnet layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
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
    CudaTensor passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
      CudaTensor cudaTensor = new CudaTensor(
          gpu.allocate(inputData.getElements() * precision.size, MemoryType.Managed.ifEnabled(), true),
          gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
      gpu.freeRef();
      return cudaTensor;
    }, inputData.addRef()));
    int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
    if (cols == 1 && rows == 1) {
      input.freeRef();
      inputData.freeRef();
      if (null != passback)
        passback.freeRef();
      assert inner != null;
      return inner.eval(inObj);
    }
    int[] tileDimensions = {width, height, bands};
    AtomicInteger counter = new AtomicInteger(0);
    Result[][] tileResults = new Result[rows][];
    for (int row = 0; row < rows; row++) {
      RefUtil.set(tileResults, row, new Result[cols]);
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX;
        int positionY = row * strideY;
        assert positionX >= 0;
        assert positionY >= 0;
        assert positionX < inputDims[0];
        assert positionY < inputDims[1];

        CudaTensor tile = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
          return ImgTileSelectLayer.copy(gpu, inputData.addRef(), inputData.getDimensions(),
              tileDimensions, precision, positionX, positionY, true);
        }, inputData.addRef()));

        assert inner != null;
        CudaTensorList data = new CudaTensorList(tile, length, tileDimensions, precision);
        Result.Accumulator accumulator = new TileAccumulator(precision, passback.addRef(), tileDimensions, positionX, positionY, counter, rows, cols, length, inputDims, input.getAccumulator());
        RefUtil.set(tileResults[row], col, inner.eval(new Result(data, accumulator)));
      }
    }
    input.freeRef();
    passback.freeRef();
    inputData.freeRef();
    logger.debug(RefString.format("Broke input %s into %s rows, %s cols", RefArrays.toString(inputDims), rows, cols));
    ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
    imgTileAssemblyLayer.setParallel(parallel);
    imgTileAssemblyLayer.setPrecision(precision);
    Result result = imgTileAssemblyLayer.eval(
        RefArrays.stream(tileResults).flatMap(array -> RefArrays.stream(array)).<Result>toArray(i -> new Result[i]));
    imgTileAssemblyLayer.freeRef();
    RefUtil.freeRef(inObj);
    assert result != null;
    TensorList data = result.getData();
    MainAccumulator accumulator = new MainAccumulator(result.getAccumulator());
    result.freeRef();
    return new Result(data, accumulator);
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

  public void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileSubnetLayer addRef() {
    return (ImgTileSubnetLayer) super.addRef();
  }

  private static class TileAccumulator extends Result.Accumulator {

    private final CudaTensor passback;
    private final int[] tileDimensions;
    private final int positionX;
    private final int positionY;
    private final AtomicInteger counter;
    private final int rows;
    private final int cols;
    private final int length;
    private final int[] inputDims;
    private Precision precision;
    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Tile accumulator.
     *
     * @param precision      the precision
     * @param passback       the passback
     * @param tileDimensions the tile dimensions
     * @param positionX      the position x
     * @param positionY      the position y
     * @param counter        the counter
     * @param rows           the rows
     * @param cols           the cols
     * @param length         the length
     * @param inputDims      the input dims
     * @param accumulator    the accumulator
     */
    public TileAccumulator(Precision precision, CudaTensor passback, int[] tileDimensions, int positionX, int positionY, AtomicInteger counter, int rows, int cols, int length, int[] inputDims, Result.Accumulator accumulator) {
      this.passback = passback;
      this.tileDimensions = tileDimensions;
      this.positionX = positionX;
      this.positionY = positionY;
      this.counter = counter;
      this.rows = rows;
      this.cols = cols;
      this.length = length;
      this.inputDims = inputDims;
      this.precision = precision;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nullable TensorList delta) {
      CudaSystem.run(RefUtil.wrapInterface((RefConsumer<CudnnHandle>) gpu -> {
        ImgTileSelectLayer.copy(gpu, delta == null ? null : delta.addRef(), tileDimensions, -positionX,
            -positionY, precision, passback == null ? null : passback.addRef());
      }, delta, passback == null ? null : passback.addRef()));
      if (counter.incrementAndGet() >= rows * cols) {
        counter.set(0);
        DeltaSet<UUID> buffer = ctx == null ? null : ctx.addRef();
        TensorList delta1 = new CudaTensorList(
            passback == null ? null : passback.addRef(), length, inputDims, precision);
        this.accumulator.accept(buffer, delta1);
      }
      if (null != ctx)
        ctx.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      passback.freeRef();
    }
  }

  private static class MainAccumulator extends Result.Accumulator {

    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Main accumulator.
     *
     * @param accumulator the accumulator
     */
    public MainAccumulator(Result.Accumulator accumulator) {
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nullable TensorList delta) {
      this.accumulator.accept(ctx, delta);
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
