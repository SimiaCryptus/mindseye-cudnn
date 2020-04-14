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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefStream;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The type Sum inputs layer.
 */
@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private boolean parallel = true;

  /**
   * Instantiates a new Sum inputs layer.
   */
  public SumInputsLayer() {
    super();
  }

  /**
   * Instantiates a new Sum inputs layer.
   *
   * @param json the json
   */
  protected SumInputsLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    setParallel(json.get("parallel").getAsBoolean());
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.SumInputsLayer();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * Is parallel boolean.
   *
   * @return the boolean
   */
  public boolean isParallel() {
    return parallel;
  }

  /**
   * Sets parallel.
   *
   * @param parallel the parallel
   */
  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  /**
   * From json sum inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum inputs layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  /**
   * Combine pipeline network.
   *
   * @param networks the networks
   * @return the pipeline network
   */
  public static PipelineNetwork combine(@Nonnull PipelineNetwork... networks) {
    if (1 == networks.length) {
      PipelineNetwork net0 = networks[0].addRef();
      RefUtil.freeRef(networks);
      return net0;
    }
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    RefUtil.freeRef(pipelineNetwork.add(new SumInputsLayer(), RefArrays.stream(networks)
        .map(RefUtil.wrapInterface((Function<? super PipelineNetwork, ? extends InnerNode>) network -> {
          return pipelineNetwork.transferNode(network, network.getHead());
        }, pipelineNetwork.addRef())).toArray(i -> new DAGNode[i])));
    return pipelineNetwork;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList in0Data = inObj[0].getData();
    @Nonnull final int[] dimensions = in0Data.getDimensions();
    in0Data.freeRef();
    if (3 != dimensions.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList temp_29_0009 = inObj[i].getData();
      int[] dimensions1 = temp_29_0009.getDimensions();
      temp_29_0009.freeRef();
      if (Tensor.length(dimensions) != Tensor.length(dimensions1)) {
        TensorList tensorList = inObj[i].getData();
        int[] tensorListDimensions = tensorList.getDimensions();
        RefUtil.freeRef(inObj);
        tensorList.freeRef();
        throw new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(tensorListDimensions));
      }
    }
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    TensorList data = fwd(RefUtil.addRef(inObj));
    Accumulator accumulator = new Accumulator(parallel, RefUtil.addRef(inObj));
    boolean alive = alive(inObj);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumInputsLayer addRef() {
    return (SumInputsLayer) super.addRef();
  }

  @NotNull
  private TensorList fwd(@Nonnull Result[] inObj) {
    RefStream<TensorList> tensorListStream = RefArrays.stream(inObj).map(x -> {
      return Result.getData(x);
    });
    if (!CoreSettings.INSTANCE().singleThreaded && parallel)
      tensorListStream = tensorListStream.parallel();
    return RefUtil.get(tensorListStream.reduce((leftData, rightData) -> {
      return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, TensorList>) gpu -> {
            TensorList tensorList = gpu.addAndFree(precision, leftData.addRef(),
                rightData == null ? null : rightData.addRef());
            gpu.freeRef();
            return tensorList;
          }, rightData == null ? null : rightData.addRef(), leftData.addRef()),
          leftData, rightData);
    }));
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  private static class Accumulator extends Result.Accumulator {

    private final Result[] inObj;
    private boolean parallel;

    /**
     * Instantiates a new Accumulator.
     *
     * @param parallel the parallel
     * @param inObj    the in obj
     */
    public Accumulator(boolean parallel, Result... inObj) {
      this.inObj = inObj;
      this.parallel = parallel;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      @Nonnull
      RefStream<Result> deltaStream = RefArrays.stream(RefUtil.addRef(inObj));
      if (!CoreSettings.INSTANCE().singleThreaded && parallel)
        deltaStream = deltaStream.parallel();
      deltaStream.filter(result -> {
        boolean alive = result.isAlive();
        result.freeRef();
        return alive;
      }).forEach(RefUtil.wrapInterface((Consumer<? super Result>) obj -> {
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        TensorList delta1 = delta == null ? null : delta.addRef();
        Result.Accumulator accumulator = obj.getAccumulator();
        try {
          accumulator.accept(buffer1, delta1);
        } finally {
          accumulator.freeRef();
        }
        obj.freeRef();
      }, buffer, delta));
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
