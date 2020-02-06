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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefSystem;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.Random;
import java.util.function.LongFunction;

@SuppressWarnings("serial")
public class StochasticSamplingSubnetLayer extends WrapperLayer
    implements StochasticComponent, MultiPrecision {

  private final int samples;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private long seed = RefSystem.nanoTime();
  private long layerSeed = RefSystem.nanoTime();

  public StochasticSamplingSubnetLayer(final Layer subnetwork, final int samples) {
    super(subnetwork);
    this.samples = samples;
  }

  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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
  public void setPrecision(Precision precision) {
    this.precision = precision;
  }

  public long[] getSeeds() {
    Random random = new Random(seed + layerSeed);
    return RefIntStream.range(0, this.samples).mapToLong(i -> random.nextLong()).toArray();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  @Nullable
  public static Result average(@Nonnull final Result[] samples, final Precision precision) {
    PipelineNetwork gateNetwork = new PipelineNetwork(1);
    ProductLayer temp_28_0006 = new ProductLayer();
    Tensor temp_28_0007 = new Tensor(1, 1, 1);
    temp_28_0006.setPrecision(precision);
    RefUtil.freeRef(gateNetwork.add(temp_28_0006, gateNetwork.getInput(0),
        gateNetwork.add(
            new ValueLayer(temp_28_0007.map(RefUtil.wrapInterface(v -> 1.0 / samples.length, RefUtil.addRefs(samples)))),
            new DAGNode[]{})));
    temp_28_0007.freeRef();
    SumInputsLayer sumInputsLayer = new SumInputsLayer();
    sumInputsLayer.setPrecision(precision);
    Result temp_28_0001 = gateNetwork.eval(sumInputsLayer.eval(samples));
    sumInputsLayer.freeRef();
    gateNetwork.freeRef();
    return temp_28_0001;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (seed == 0) {
      Layer temp_28_0009 = getInner();
      assert temp_28_0009 != null;
      Result temp_28_0005 = temp_28_0009.eval(inObj);
      temp_28_0009.freeRef();
      return temp_28_0005;
    }
    Result[] counting = RefArrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(i -> new Result[i]);
    return average(
        RefArrays.stream(getSeeds()).mapToObj(RefUtil.wrapInterface((LongFunction<? extends Result>) seed1 -> {
          Layer inner = getInner();
          if (inner instanceof DAGNetwork) {
            ((DAGNetwork) inner).visitNodes(node -> {
              Layer layer = node.getLayer();
              node.freeRef();
              if (layer instanceof StochasticComponent) {
                ((StochasticComponent) layer).shuffle(seed1);
              }
              if (layer instanceof MultiPrecision) {
                ((MultiPrecision) layer).setPrecision(precision);
              }
              if (null != layer)
                layer.freeRef();
            });
          }
          if (inner instanceof MultiPrecision) {
            ((MultiPrecision) inner).setPrecision(precision);
          }
          if (inner instanceof StochasticComponent) {
            ((StochasticComponent) inner).shuffle(seed1);
          }
          assert inner != null;
          inner.setFrozen(isFrozen());
          Result temp_28_0004 = inner.eval(RefUtil.addRefs(counting));
          inner.freeRef();
          return temp_28_0004;
        }, counting)).toArray(i -> new Result[i]), precision);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  StochasticSamplingSubnetLayer addRef() {
    return (StochasticSamplingSubnetLayer) super.addRef();
  }
}
