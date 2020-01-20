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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public class ImgModulusCropLayer extends LayerBase implements MultiPrecision {
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

  public int getOffsetX() {
    return offsetX;
  }

  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public boolean isRoundUp() {
    return roundUp;
  }

  public void setRoundUp(boolean roundUp) {
    this.roundUp = roundUp;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgModulusCropLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgModulusCropLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    TensorList temp_40_0004 = inObj[0].getData();
    @Nonnull
    int[] dimensions = temp_40_0004.getDimensions();
    temp_40_0004.freeRef();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];

    int sizeX = Math.abs(this.sizeX);
    int paddingX = sizeX - ((inputWidth - offsetX) % sizeX);
    while (paddingX < 0)
      paddingX += sizeX;
    while (paddingX >= sizeX)
      paddingX -= sizeX;
    if (this.sizeX < 0 && (paddingX + inputWidth) > sizeX)
      paddingX -= sizeX;
    while (paddingX > 0)
      paddingX -= sizeX;

    int sizeY = Math.abs(this.sizeY);
    int paddingY = sizeY - ((inputHeight - offsetY) % sizeY);
    while (paddingY < 0)
      paddingY += sizeY;
    while (paddingY >= sizeY)
      paddingY -= sizeY;
    if (this.sizeY < 0 && (paddingY + inputHeight) > sizeY)
      paddingY -= sizeY;
    while (paddingY > 0)
      paddingY -= sizeY;

    int ouputWidth = inputWidth + paddingX;
    int outputHeight = inputHeight + paddingY;
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        Result temp_40_0002 = inObj[0].addRef();
        ReferenceCounting.freeRefs(inObj);
        return temp_40_0002;
      }
    }

    ImgCropLayer temp_40_0003 = new ImgCropLayer(ouputWidth, outputHeight);
    temp_40_0003.setPrecision(precision);
    ImgCropLayer temp_40_0005 = RefUtil.addRef(temp_40_0003);
    temp_40_0005.setRoundUp(isRoundUp());
    @Nonnull
    ImgCropLayer imgCropLayer = temp_40_0005.addRef();
    temp_40_0005.freeRef();
    temp_40_0003.freeRef();
    Result temp_40_0001 = imgCropLayer.eval(RefUtil.addRefs(inObj));
    ReferenceCounting.freeRefs(inObj);
    imgCropLayer.freeRef();
    return temp_40_0001;
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
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgModulusCropLayer addRef() {
    return (ImgModulusCropLayer) super.addRef();
  }
}
