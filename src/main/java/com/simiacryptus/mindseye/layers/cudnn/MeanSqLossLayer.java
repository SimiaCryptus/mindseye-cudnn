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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class MeanSqLossLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  private final InnerNode binaryNode;
  private double alpha = 1.0;

  public MeanSqLossLayer() {
    super(2);
    final Layer nextHead = new BinarySumLayer(alpha, -alpha);
    this.binaryNode = add(nextHead, getInput(0), getInput(1));
    add(new SquareActivationLayer());
    add(new AvgReducerLayer());
  }

  protected MeanSqLossLayer(@Nonnull final JsonObject id,
                            Map<CharSequence, byte[]> rs) {
    super(id, rs);
    alpha = id.get("alpha").getAsDouble();
    binaryNode = (InnerNode) getNodeById(UUID.fromString(id.get("binaryNode").getAsString()));
  }

  public double getAlpha() {
    return alpha;
  }

  public MeanSqLossLayer setAlpha(final double alpha) {
    this.alpha = alpha;
    BinarySumLayer binarySumLayer = binaryNode.getLayer();
    binarySumLayer.setLeftFactor(alpha);
    binarySumLayer.setRightFactor(-alpha);
    return this;
  }

  @SuppressWarnings("unused")
  public static MeanSqLossLayer fromJson(@NotNull final JsonObject json,
                                         Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  MeanSqLossLayer[] addRefs(MeanSqLossLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayer::addRef)
        .toArray((x) -> new MeanSqLossLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MeanSqLossLayer[][] addRefs(MeanSqLossLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayer::addRefs)
        .toArray((x) -> new MeanSqLossLayer[x][]);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("alpha", alpha);
    json.addProperty("binaryNode", binaryNode.id.toString());
    return json;
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  MeanSqLossLayer addRef() {
    return (MeanSqLossLayer) super.addRef();
  }
}
