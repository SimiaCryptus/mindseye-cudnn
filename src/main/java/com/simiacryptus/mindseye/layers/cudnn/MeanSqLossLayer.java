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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Mean sq loss layer.
 */
@SuppressWarnings("serial")
public class MeanSqLossLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  @Nullable
  private final InnerNode binaryNode;
  private double alpha = 1.0;

  /**
   * Instantiates a new Mean sq loss layer.
   */
  public MeanSqLossLayer() {
    super(2);
    this.binaryNode = add(new BinarySumLayer(alpha, -alpha), getInput(0), getInput(1));
    RefUtil.freeRef(add(new SquareActivationLayer()));
    RefUtil.freeRef(add(new AvgReducerLayer()));
  }

  /**
   * Instantiates a new Mean sq loss layer.
   *
   * @param id the id
   * @param rs the rs
   */
  protected MeanSqLossLayer(@Nonnull final JsonObject id, Map<CharSequence, byte[]> rs) {
    super(id, rs);
    alpha = id.get("alpha").getAsDouble();
    binaryNode = (InnerNode) getNodeById(UUID.fromString(id.get("binaryNode").getAsString()));
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   */
  public void setAlpha(double alpha) {
    this.alpha = alpha;
    assert binaryNode != null;
    BinarySumLayer binarySumLayer = binaryNode.getLayer();
    binarySumLayer.setLeftFactor(alpha);
    binarySumLayer.setRightFactor(-alpha);
    binarySumLayer.freeRef();
  }

  /**
   * From json mean sq loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the mean sq loss layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static MeanSqLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json, rs);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    assert json != null;
    json.addProperty("alpha", alpha);
    assert binaryNode != null;
    json.addProperty("binaryNode", binaryNode.id.toString());
    return json;
  }

  public void _free() {
    if (null != binaryNode)
      binaryNode.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MeanSqLossLayer addRef() {
    return (MeanSqLossLayer) super.addRef();
  }
}
