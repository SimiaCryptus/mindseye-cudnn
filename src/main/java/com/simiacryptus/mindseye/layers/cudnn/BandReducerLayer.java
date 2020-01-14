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
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public class BandReducerLayer extends LayerBase implements MultiPrecision<BandReducerLayer> {

  private PoolingLayer.PoolingMode mode = PoolingLayer.PoolingMode.Max;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private double alpha = 1.0;

  public BandReducerLayer() {
    super();
  }

  protected BandReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = RefUtil.get(RefArrays.stream(PoolingLayer.PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt())
        .findFirst());
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }

  public double getAlpha() {
    return alpha;
  }

  @Nonnull
  public BandReducerLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this.addRef();
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }

  public PoolingMode getMode() {
    return mode;
  }

  @Nonnull
  public BandReducerLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public BandReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BandReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandReducerLayer(json);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BandReducerLayer[] addRefs(@Nullable BandReducerLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandReducerLayer::addRef)
        .toArray((x) -> new BandReducerLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BandReducerLayer[][] addRefs(@Nullable BandReducerLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandReducerLayer::addRefs)
        .toArray((x) -> new BandReducerLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nullable final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_27_0004 = getCompatibilityLayer();
      Result temp_27_0002 = temp_27_0004.eval(Result.addRefs(inObj));
      temp_27_0004.freeRef();
      if (null != inObj)
        ReferenceCounting.freeRefs(inObj);
      return temp_27_0002;
    }
    assert inObj != null;
    final Result input = inObj[0].addRef();
    final TensorList batch = input.getData();
    input.freeRef();
    @Nonnull final int[] inputSize = batch.getDimensions();
    batch.freeRef();
    PoolingLayer temp_27_0003 = new PoolingLayer();
    PoolingLayer temp_27_0005 = temp_27_0003.setMode(mode);
    PoolingLayer temp_27_0006 = temp_27_0005.setPrecision(precision);
    PoolingLayer temp_27_0007 = temp_27_0006.setWindowX(inputSize[0]);
    PoolingLayer temp_27_0008 = temp_27_0007.setWindowY(inputSize[1]);
    PoolingLayer temp_27_0009 = temp_27_0008.setStrideX(inputSize[0]);
    PoolingLayer temp_27_0010 = temp_27_0009.setStrideY(inputSize[1]);
    PoolingLayer temp_27_0011 = temp_27_0010.setPaddingX(0);
    PoolingLayer temp_27_0012 = temp_27_0011.setPaddingY(0);
    @Nonnull
    PoolingLayer impl = temp_27_0012.setAlpha(alpha);
    temp_27_0012.freeRef();
    temp_27_0011.freeRef();
    temp_27_0010.freeRef();
    temp_27_0009.freeRef();
    temp_27_0008.freeRef();
    temp_27_0007.freeRef();
    temp_27_0006.freeRef();
    temp_27_0005.freeRef();
    temp_27_0003.freeRef();
    Result temp_27_0001 = impl.eval(Result.addRefs(inObj));
    ReferenceCounting.freeRefs(inObj);
    impl.freeRef();
    return temp_27_0001;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", alpha);
    json.addProperty("mode", mode.id);
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
  BandReducerLayer addRef() {
    return (BandReducerLayer) super.addRef();
  }
}
