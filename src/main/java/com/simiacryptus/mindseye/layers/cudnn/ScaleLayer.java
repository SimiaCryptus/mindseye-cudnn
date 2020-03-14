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
import com.simiacryptus.ref.lang.RefUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;

@SuppressWarnings("serial")
public class ScaleLayer extends PipelineNetwork implements MultiPrecision {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleLayer.class);

  public ScaleLayer() {
    this(new Tensor(1));
  }

  public ScaleLayer(double value) {
    this(new Tensor(new double[]{value}, 1));
  }

  public ScaleLayer(@Nullable final Tensor weights) {
    super(1);
    RefUtil.freeRef(add(new ProductLayer(), getInput(0), add(new ValueLayer(weights), new DAGNode[]{})));
  }

  protected ScaleLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    //weights = new Tensor(1);
  }

  @Nullable
  @Override
  public Precision getPrecision() {
    return null;
  }

  @Override
  public void setPrecision(Precision precision) {
    MultiPrecision.setPrecision(this.addRef(), precision);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ScaleLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleLayer(json, rs);
  }


  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ScaleLayer addRef() {
    return (ScaleLayer) super.addRef();
  }
}
