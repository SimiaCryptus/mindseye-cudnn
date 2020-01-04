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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.mindseye.network.CountingResult;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class StochasticSamplingSubnetLayer extends WrapperLayer
    implements StochasticComponent, MultiPrecision<StochasticSamplingSubnetLayer> {

  private final int samples;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private long seed = System.nanoTime();
  private long layerSeed = System.nanoTime();

  public StochasticSamplingSubnetLayer(final Layer subnetwork, final int samples) {
    super(subnetwork);
    this.samples = samples;
  }

  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json,
                                          com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsInt();
    layerSeed = json.getAsJsonPrimitive("layerSeed").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public StochasticSamplingSubnetLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }

  public long[] getSeeds() {
    Random random = new Random(seed + layerSeed);
    return com.simiacryptus.ref.wrappers.RefIntStream.range(0, this.samples).mapToLong(i -> random.nextLong())
        .toArray();
  }

  @SuppressWarnings("unused")
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json,
                                                       com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  public static Result average(final Result[] samples, final Precision precision) {
    PipelineNetwork gateNetwork = new PipelineNetwork(1);
    gateNetwork.add(new ProductLayer().setPrecision(precision), gateNetwork.getInput(0),
        gateNetwork.add(new ValueLayer(new Tensor(1, 1, 1).map(v -> 1.0 / samples.length)), new DAGNode[]{}));
    SumInputsLayer sumInputsLayer = new SumInputsLayer().setPrecision(precision);
    return gateNetwork.eval(sumInputsLayer.eval(samples));
  }

  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer[] addRefs(
      StochasticSamplingSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRef)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer[][] addRefs(
      StochasticSamplingSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayer::addRefs)
        .toArray((x) -> new StochasticSamplingSubnetLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (seed == 0) {
      return getInner().eval(inObj);
    }
    Result[] counting = com.simiacryptus.ref.wrappers.RefArrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(i -> new Result[i]);
    return average(com.simiacryptus.ref.wrappers.RefArrays.stream(getSeeds()).mapToObj(seed -> {
      Layer inner = getInner();
      if (inner instanceof DAGNetwork) {
        ((DAGNetwork) inner).visitNodes(node -> {
          Layer layer = node.getLayer();
          if (layer instanceof StochasticComponent) {
            ((StochasticComponent) layer).shuffle(seed);
          }
          if (layer instanceof MultiPrecision<?>) {
            ((MultiPrecision) layer).setPrecision(precision);
          }
        });
      }
      if (inner instanceof MultiPrecision<?>) {
        ((MultiPrecision) inner).setPrecision(precision);
      }
      if (inner instanceof StochasticComponent) {
        ((StochasticComponent) inner).shuffle(seed);
      }
      inner.setFrozen(isFrozen());
      return inner.eval(counting);
    }).toArray(i -> new Result[i]), precision);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("samples", samples);
    json.addProperty("seed", seed);
    json.addProperty("layerSeed", layerSeed);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Override
  public void shuffle(final long seed) {
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    seed = 0;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer addRef() {
    return (StochasticSamplingSubnetLayer) super.addRef();
  }
}
