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
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class RescaledSubnetLayer extends LayerBase
    implements MultiPrecision<RescaledSubnetLayer> {
  private static final Logger log = LoggerFactory.getLogger(RescaledSubnetLayer.class);

  private int scale;
  private Layer layer;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private RescaledSubnetLayer() {
  }

  public RescaledSubnetLayer(int scale, Layer layer) {
    this.scale = scale;
    this.layer = layer;
  }

  protected RescaledSubnetLayer(@Nonnull final JsonObject json,
                                com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    scale = json.get("scale").getAsInt();
    layer = Layer.fromJson(json, rs);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer(scale, layer);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public RescaledSubnetLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static RescaledSubnetLayer fromJson(@Nonnull final JsonObject json,
                                             com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayer[] addRefs(RescaledSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRef)
        .toArray((x) -> new RescaledSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayer[][] addRefs(RescaledSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRefs)
        .toArray((x) -> new RescaledSubnetLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    log.warn("Not Implemented: " + getClass().getCanonicalName());
    return getCompatibilityLayer().eval(inObj);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    json.add("key", layer.getJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  RescaledSubnetLayer addRef() {
    return (RescaledSubnetLayer) super.addRef();
  }
}
