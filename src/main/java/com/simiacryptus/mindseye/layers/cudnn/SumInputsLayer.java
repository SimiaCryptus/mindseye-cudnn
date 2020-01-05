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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;

@SuppressWarnings("serial")
public @RefAware
class SumInputsLayer extends LayerBase implements MultiPrecision<SumInputsLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public SumInputsLayer() {
    super();
  }

  protected SumInputsLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    RefUtil.freeRef(setParallel(json.get("parallel").getAsBoolean()));
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.SumInputsLayer();

  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SumInputsLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public boolean isParallel() {
    return parallel;
  }

  public SumInputsLayer setParallel(boolean parallel) {
    this.parallel = parallel;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(PipelineNetwork... networks) {
    if (1 == networks.length) {
      PipelineNetwork temp_29_0004 = networks[0];
      if (null != networks)
        ReferenceCounting.freeRefs(networks);
      return temp_29_0004;
    }
    RefArrays.stream(PipelineNetwork.addRefs(networks))
        .forEach(ReferenceCountingBase::assertAlive);
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    RefUtil.freeRef(pipelineNetwork.add(new SumInputsLayer(), RefArrays
        .stream(PipelineNetwork.addRefs(networks))
        .map(RefUtil.wrapInterface(
            (Function<? super PipelineNetwork, ? extends InnerNode>) network -> {
              InnerNode temp_29_0001 = PipelineNetwork
                  .transferNode(pipelineNetwork == null ? null : pipelineNetwork.addRef(), network.getHead());
              if (null != network)
                network.freeRef();
              return temp_29_0001;
            }, pipelineNetwork == null ? null : pipelineNetwork.addRef()))
        .toArray(i -> new DAGNode[i])));
    if (null != networks)
      ReferenceCounting.freeRefs(networks);
    return pipelineNetwork;
  }

  public static @SuppressWarnings("unused")
  SumInputsLayer[] addRefs(SumInputsLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayer::addRef)
        .toArray((x) -> new SumInputsLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SumInputsLayer[][] addRefs(SumInputsLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayer::addRefs)
        .toArray((x) -> new SumInputsLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_29_0008 = inObj[0].getData();
    @Nonnull final int[] dimensions = temp_29_0008.getDimensions();
    if (null != temp_29_0008)
      temp_29_0008.freeRef();
    if (3 != dimensions.length) {
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList temp_29_0009 = inObj[i].getData();
      if (Tensor.length(dimensions) != Tensor.length(temp_29_0009.getDimensions())) {
        TensorList temp_29_0010 = inObj[i].getData();
        IllegalArgumentException temp_29_0006 = new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(temp_29_0010.getDimensions()));
        if (null != temp_29_0010)
          temp_29_0010.freeRef();
        if (null != inObj)
          ReferenceCounting.freeRefs(inObj);
        throw temp_29_0006;
      }
      if (null != temp_29_0009)
        temp_29_0009.freeRef();
    }
    if (!CudaSystem.isEnabled()) {
      Layer temp_29_0011 = getCompatibilityLayer();
      Result temp_29_0007 = temp_29_0011
          .eval(Result.addRefs(inObj));
      if (null != temp_29_0011)
        temp_29_0011.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_29_0007;
    }
    RefStream<TensorList> tensorListStream = RefArrays.stream(Result.addRefs(inObj))
        .map(x -> {
          TensorList temp_29_0002 = x.getData();
          if (null != x)
            x.freeRef();
          return temp_29_0002;
        });
    if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
      tensorListStream = tensorListStream.parallel();
    try {
      return new Result(tensorListStream.reduce((leftData, rightData) -> {
        TensorList temp_29_0003 = CudaSystem
            .run(RefUtil.wrapInterface(
                (Function<CudnnHandle, TensorList>) gpu -> {
                  return gpu.addAndFree(precision, leftData == null ? null : leftData.addRef(),
                      rightData == null ? null : rightData.addRef());
                }, rightData == null ? null : rightData.addRef(), leftData == null ? null : leftData.addRef()),
                leftData == null ? null : leftData.addRef(), rightData == null ? null : rightData.addRef());
        if (null != rightData)
          rightData.freeRef();
        if (null != leftData)
          leftData.freeRef();
        return temp_29_0003;
      }).get(), new Result.Accumulator() {
        {
          Result.addRefs(inObj);
        }

        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList delta) {
          @Nonnull
          RefStream<Result> deltaStream = RefArrays.stream(Result.addRefs(inObj));
          if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
            deltaStream = deltaStream.parallel();
          deltaStream.filter(Result::isAlive).forEach(RefUtil
              .wrapInterface((Consumer<? super Result>) obj -> {
                obj.accumulate(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                if (null != obj)
                  obj.freeRef();
              }, buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef()));
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      }) {

        {
          Result.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          for (@Nonnull final Result element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
        }

      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
    }
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
  }

  public @Override
  @SuppressWarnings("unused")
  SumInputsLayer addRef() {
    return (SumInputsLayer) super.addRef();
  }
}
