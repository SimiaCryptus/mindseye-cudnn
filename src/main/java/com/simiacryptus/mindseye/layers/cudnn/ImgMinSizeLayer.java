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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgMinSizeLayer extends LayerBase
    implements MultiPrecision<ImgMinSizeLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgMinSizeLayer.class);

  private int sizeX;
  private int sizeY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private ImgMinSizeLayer() {
  }

  public ImgMinSizeLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  protected ImgMinSizeLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgMinSizeLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgMinSizeLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgMinSizeLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    Result in0 = inObj[0];
    @Nonnull
    int[] dimensions = in0.getData().getDimensions();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];

    int ouputWidth = Math.max(inputWidth, sizeX);
    int outputHeight = Math.max(inputHeight, sizeY);
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        return in0;
      }
    }

    @Nonnull
    ImgCropLayer imgCropLayer = new ImgCropLayer(ouputWidth, outputHeight).setPrecision(precision);
    return imgCropLayer.eval(inObj);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgMinSizeLayer addRef() {
    return (ImgMinSizeLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgMinSizeLayer[] addRefs(ImgMinSizeLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgMinSizeLayer::addRef)
        .toArray((x) -> new ImgMinSizeLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgMinSizeLayer[][] addRefs(ImgMinSizeLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgMinSizeLayer::addRefs)
        .toArray((x) -> new ImgMinSizeLayer[x][]);
  }

}
