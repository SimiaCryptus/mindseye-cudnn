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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class ScaleLayer extends PipelineNetwork
    implements MultiPrecision<ScaleLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleLayer.class);

  public ScaleLayer() {
    this(new Tensor(1));
  }

  public ScaleLayer(double value) {
    this(new Tensor(1).setAll(value));
  }

  public ScaleLayer(final Tensor weights) {
    super(1);
    add(new ProductLayer(), getInput(0), add(new ValueLayer(weights), new DAGNode[]{}));
  }

  protected ScaleLayer(@Nonnull final JsonObject json, com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
    //weights = new Tensor(1);
  }

  @Override
  public Precision getPrecision() {
    return null;
  }

  @Nonnull
  @Override
  public ScaleLayer setPrecision(Precision precision) {
    return MultiPrecision.setPrecision(this, precision);
  }

  @SuppressWarnings("unused")
  public static ScaleLayer fromJson(@NotNull final JsonObject json,
                                    com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ScaleLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ScaleLayer[] addRefs(ScaleLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ScaleLayer::addRef)
        .toArray((x) -> new ScaleLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ScaleLayer[][] addRefs(ScaleLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ScaleLayer::addRefs)
        .toArray((x) -> new ScaleLayer[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ScaleLayer addRef() {
    return (ScaleLayer) super.addRef();
  }
}
