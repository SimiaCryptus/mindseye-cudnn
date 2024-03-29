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
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;

/**
 * The type Img min size layer.
 */
@SuppressWarnings("serial")
public class ImgMinSizeLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgMinSizeLayer.class);

  private int sizeX;
  private int sizeY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  private ImgMinSizeLayer() {
  }

  /**
   * Instantiates a new Img min size layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgMinSizeLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  /**
   * Instantiates a new Img min size layer.
   *
   * @param json the json
   */
  protected ImgMinSizeLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * From json img min size layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img min size layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgMinSizeLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgMinSizeLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    Result in0 = inObj[0].addRef();
    TensorList temp_45_0003 = in0.getData();
    @Nonnull
    int[] dimensions = temp_45_0003.getDimensions();
    temp_45_0003.freeRef();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];

    int ouputWidth = Math.max(inputWidth, sizeX);
    int outputHeight = Math.max(inputHeight, sizeY);
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        RefUtil.freeRef(inObj);
        return in0;
      }
    }

    in0.freeRef();
    ImgCropLayer cropLayer = new ImgCropLayer(ouputWidth, outputHeight);
    cropLayer.setPrecision(precision);
    Result result = cropLayer.eval(inObj);
    cropLayer.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgMinSizeLayer addRef() {
    return (ImgMinSizeLayer) super.addRef();
  }

}
