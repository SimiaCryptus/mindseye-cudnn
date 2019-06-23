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

@SuppressWarnings("serial")
public class ImgModulusCropLayer extends LayerBase implements MultiPrecision<ImgModulusCropLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgModulusCropLayer.class);
  private boolean roundUp = false;

  private int sizeX;
  private int sizeY;
  private int offsetX;
  private int offsetY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private ImgModulusCropLayer() {
  }

  public ImgModulusCropLayer(int sizeX, int sizeY, int offsetX, int offsetY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.offsetX = offsetX;
    this.offsetY = offsetY;
  }

  public ImgModulusCropLayer(int sizeX, int sizeY) {
    this(sizeX, sizeY, 0, 0);
  }

  protected ImgModulusCropLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    setRoundUp(json.get("roundUp").getAsBoolean());
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    offsetX = json.get("offsetX").getAsInt();
    offsetY = json.get("offsetY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  public static ImgModulusCropLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgModulusCropLayer(json, rs);
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
    while (paddingX > 0) paddingX -= sizeX;

    int sizeY = Math.abs(this.sizeY);
    int paddingY = sizeY - ((inputHeight - offsetY) % sizeY);
    while (paddingY < 0) paddingY += sizeY;
    while (paddingY >= sizeY) paddingY -= sizeY;
    if (this.sizeY < 0 && (paddingY + inputHeight) > sizeY) paddingY -= sizeY;
    while (paddingY > 0) paddingY -= sizeY;

    int ouputWidth = inputWidth + paddingX;
    int outputHeight = inputHeight + paddingY;
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        return inObj[0];
      }
    }

    @Nonnull ImgCropLayer imgCropLayer = new ImgCropLayer(ouputWidth, outputHeight).setPrecision(precision).setRoundUp(isRoundUp());
    @Nullable Result eval = imgCropLayer.evalAndFree(inObj);
    imgCropLayer.freeRef();
    return eval;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("roundUp", roundUp);
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
  public ImgModulusCropLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public int getOffsetX() {
    return offsetX;
  }

  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public ImgModulusCropLayer setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
    return this;
  }
}
