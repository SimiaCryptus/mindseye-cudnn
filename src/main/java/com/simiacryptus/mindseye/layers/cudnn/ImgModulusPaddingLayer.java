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
 * The type Img modulus padding layer.
 */
@SuppressWarnings("serial")
public class ImgModulusPaddingLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(ImgModulusPaddingLayer.class);

  private int sizeX;
  private int sizeY;
  private int offsetX;
  private int offsetY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private boolean roundUp = false;

  private ImgModulusPaddingLayer() {
  }

  /**
   * Instantiates a new Img modulus padding layer.
   *
   * @param sizeX   the size x
   * @param sizeY   the size y
   * @param offsetX the offset x
   * @param offsetY the offset y
   */
  public ImgModulusPaddingLayer(int sizeX, int sizeY, int offsetX, int offsetY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.offsetX = offsetX;
    this.offsetY = offsetY;
  }

  /**
   * Instantiates a new Img modulus padding layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgModulusPaddingLayer(int sizeX, int sizeY) {
    this(sizeX, sizeY, 0, 0);
  }

  /**
   * Instantiates a new Img modulus padding layer.
   *
   * @param json the json
   */
  protected ImgModulusPaddingLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    roundUp = json.get("roundUp").getAsBoolean();
    offsetX = json.get("offsetX").getAsInt();
    offsetY = json.get("offsetY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  /**
   * Gets offset x.
   *
   * @return the offset x
   */
  public int getOffsetX() {
    return offsetX;
  }

  /**
   * Sets offset x.
   *
   * @param offsetX the offset x
   */
  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
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
   * Gets round up.
   *
   * @return the round up
   */
  public boolean getRoundUp() {
    return roundUp;
  }

  /**
   * Sets round up.
   *
   * @param roundUp the round up
   */
  public void setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
  }

  /**
   * From json img modulus padding layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img modulus padding layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgModulusPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgModulusPaddingLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    TensorList temp_44_0004 = inObj[0].getData();
    @Nonnull
    int[] dimensions = temp_44_0004.getDimensions();
    temp_44_0004.freeRef();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];

    int sizeX = Math.abs(this.sizeX);
    int paddingX = sizeX - (inputWidth - offsetX) % sizeX;
    while (paddingX < 0)
      paddingX += sizeX;
    while (paddingX >= sizeX)
      paddingX -= sizeX;
    if (this.sizeX < 0 && paddingX + inputWidth > sizeX)
      paddingX -= sizeX;

    int sizeY = Math.abs(this.sizeY);
    int paddingY = sizeY - (inputHeight - offsetY) % sizeY;
    while (paddingY < 0)
      paddingY += sizeY;
    while (paddingY >= sizeY)
      paddingY -= sizeY;
    if (this.sizeY < 0 && paddingY + inputHeight > sizeY)
      paddingY -= sizeY;

    int ouputWidth = inputWidth + paddingX;
    int outputHeight = inputHeight + paddingY;
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        Result temp_44_0002 = inObj[0].addRef();
        RefUtil.freeRef(inObj);
        return temp_44_0002;
      }
    }

    ImgCropLayer imgCropLayer = new ImgCropLayer(ouputWidth, outputHeight);
    imgCropLayer.setPrecision(precision);
    imgCropLayer.setRoundUp(roundUp);
    Result result = imgCropLayer.eval(inObj);
    imgCropLayer.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("roundUp", roundUp);
    json.addProperty("sizeX", sizeX);
    json.addProperty("offsetX", offsetX);
    json.addProperty("offsetY", offsetY);
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
  ImgModulusPaddingLayer addRef() {
    return (ImgModulusPaddingLayer) super.addRef();
  }
}
