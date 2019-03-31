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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Increases the resolution of the input by selecting a larger centered window. The output png will have the same
 * number of color bands, and the area outside the source png will be setWeights to 0.
 */
@SuppressWarnings("serial")
public class ImgModulusPaddingSubnetLayer extends WrapperLayer implements MultiPrecision<ImgModulusPaddingSubnetLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgModulusPaddingSubnetLayer.class);

  private int sizeX;
  private int sizeY;
  private int offsetX;
  private int offsetY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  /**
   * Instantiates a new Img eval key.
   */
  private ImgModulusPaddingSubnetLayer() {
  }

  /**
   * Instantiates a new Img zero padding key.
   *
   * @param sizeX   the size x
   * @param sizeY   the size y
   * @param offsetX the offset x
   * @param offsetY the offset y
   * @param inner
   */
  public ImgModulusPaddingSubnetLayer(int sizeX, int sizeY, int offsetX, int offsetY, Layer inner) {
    super(inner);
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.offsetX = offsetX;
    this.offsetY = offsetY;
  }

  /**
   * Instantiates a new Img modulus padding key.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   * @param inner
   */
  public ImgModulusPaddingSubnetLayer(int sizeX, int sizeY, Layer inner) {
    this(sizeX, sizeY, 0, 0, inner);
  }

  /**
   * Instantiates a new Img eval key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgModulusPaddingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    offsetX = json.get("offsetX").getAsInt();
    offsetY = json.get("offsetY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  /**
   * From json img eval key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval key
   */
  public static ImgModulusPaddingSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgModulusPaddingSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    @Nonnull int[] dimensions = inObj[0].getData().getDimensions();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];

    int sizeX = Math.abs(this.sizeX);
    int paddingX = sizeX - ((inputWidth - offsetX) % sizeX);
    while (paddingX < 0) paddingX += sizeX;
    while (paddingX >= sizeX) paddingX -= sizeX;
    if (this.sizeX < 0 && (paddingX + inputWidth) > sizeX) paddingX -= sizeX;

    int sizeY = Math.abs(this.sizeY);
    int paddingY = sizeY - ((inputHeight - offsetY) % sizeY);
    while (paddingY < 0) paddingY += sizeY;
    while (paddingY >= sizeY) paddingY -= sizeY;
    if (this.sizeY < 0 && (paddingY + inputHeight) > sizeY) paddingY -= sizeY;

    int ouputWidth = inputWidth + paddingX;
    int outputHeight = inputHeight + paddingY;
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        return inObj[0];
      }
    }

    @Nonnull ImgCropLayer imgCropLayer1 = new ImgCropLayer(ouputWidth, outputHeight).setPrecision(precision);
    @Nonnull ImgCropLayer imgCropLayer2 = new ImgCropLayer(inputWidth, inputHeight).setPrecision(precision);
    @Nullable Result eval = imgCropLayer2.evalAndFree(super.evalAndFree(imgCropLayer1.evalAndFree(inObj)));
    imgCropLayer1.freeRef();
    imgCropLayer2.freeRef();

    return eval;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("offsetX", offsetX);
    json.addProperty("offsetY", offsetY);
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
  public ImgModulusPaddingSubnetLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
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
}
