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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
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
    mode = Arrays.stream(PoolingLayer.PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst()
        .get();
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }

  public double getAlpha() {
    return alpha;
  }

  public BandReducerLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
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
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public BandReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static BandReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandReducerLayer(json);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputSize = batch.getDimensions();
    @Nonnull
    PoolingLayer impl = new PoolingLayer().setMode(mode).setPrecision(precision).setWindowX(inputSize[0])
        .setWindowY(inputSize[1]).setStrideX(inputSize[0]).setStrideY(inputSize[1]).setPaddingX(0).setPaddingY(0)
        .setAlpha(alpha);
    return impl.eval(inObj);
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
  public List<double[]> state() {
    return Arrays.asList();
  }
}
