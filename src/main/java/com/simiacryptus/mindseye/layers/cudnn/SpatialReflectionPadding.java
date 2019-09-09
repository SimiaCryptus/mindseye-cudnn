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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ImgCropLayer.Alignment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@SuppressWarnings("serial")
public class SpatialReflectionPadding extends LayerBase implements MultiPrecision<SpatialReflectionPadding> {
  private static final Logger log = LoggerFactory.getLogger(SpatialReflectionPadding.class);
  private Alignment verticalAlign = Alignment.Center;
  private Alignment horizontalAlign = Alignment.Center;
  private boolean roundUp = false;
  private int sizeX;
  private int sizeY; // SpatialReflectionPadding
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private SpatialReflectionPadding() {
  }

  public SpatialReflectionPadding(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  protected SpatialReflectionPadding(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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

  public static SpatialReflectionPadding fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SpatialReflectionPadding(json, rs);
  }


  public Alignment getVerticalAlign() {
    return verticalAlign;
  }

  public SpatialReflectionPadding setVerticalAlign(Alignment verticalAlign) {
    this.verticalAlign = verticalAlign;
    return this;
  }

  public Alignment getHorizontalAlign() {
    return horizontalAlign;
  }

  public SpatialReflectionPadding setHorizontalAlign(Alignment horizontalAlign) {
    this.horizontalAlign = horizontalAlign;
    return this;
  }

  public int half(int i, Alignment alignment) {
    if (alignment == Alignment.Left) return 0;
    if (alignment == Alignment.Right) return i;
    if (i % 2 == 0) return i / 2;
    else if (isRoundUp()) return (i + 1) / 2;
    else return (i - 1) / 2;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (inObj.length != 1) throw new IllegalArgumentException();
    final int[] dimensions = inObj[0].getData().getDimensions();
    final ImgPaddingLayer paddingLayer = new ImgPaddingLayer(dimensions[0] + sizeX, dimensions[1] + sizeY)
        .setHorizontalAlign(horizontalAlign).setVerticalAlign(verticalAlign);
    final Result result = paddingLayer.evalAndFree(inObj);
    paddingLayer.freeRef();
    return result;
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
  public List<double[]> state() {
    return Arrays.asList();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SpatialReflectionPadding setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public SpatialReflectionPadding setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
    return this;
  }

}
