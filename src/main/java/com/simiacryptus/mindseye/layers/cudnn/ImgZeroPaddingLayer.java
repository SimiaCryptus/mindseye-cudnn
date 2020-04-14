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
 * The type Img zero padding layer.
 */
@SuppressWarnings("serial")
public class ImgZeroPaddingLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgZeroPaddingLayer.class);
  //  public StackTraceElement[] createdBy = Thread.currentThread().getStackTrace();
  private int sizeX;
  private int sizeY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  private ImgZeroPaddingLayer() {
  }

  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgZeroPaddingLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert sizeY != 0 || sizeX != 0;
  }

  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param json the json
   */
  protected ImgZeroPaddingLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    assert sizeY != 0 || sizeX != 0;
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
   * From json img zero padding layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img zero padding layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgZeroPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgZeroPaddingLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (sizeX == 0 && sizeY == 0) {
      Result temp_22_0002 = inObj[0].addRef();
      RefUtil.freeRef(inObj);
      return temp_22_0002;
    }
    assert inObj.length == 1;
    TensorList temp_22_0004 = inObj[0].getData();
    @Nonnull
    int[] dimensions = temp_22_0004.getDimensions();
    temp_22_0004.freeRef();
    ImgCropLayer cropLayer = new ImgCropLayer(dimensions[0] + 2 * this.sizeX, dimensions[1] + 2 * this.sizeY);
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
  ImgZeroPaddingLayer addRef() {
    return (ImgZeroPaddingLayer) super.addRef();
  }

}
