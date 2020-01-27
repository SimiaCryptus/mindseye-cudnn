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
public class SumInputsLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public SumInputsLayer() {
    super();
  }

  protected SumInputsLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    setParallel(json.get("parallel").getAsBoolean());
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
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public boolean isParallel() {
    return parallel;
  }

  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(@Nonnull PipelineNetwork... networks) {
    if (1 == networks.length) {
      PipelineNetwork temp_29_0004 = networks[0].addRef();
      RefUtil.freeRefs(networks);
      return temp_29_0004;
    }
    RefArrays.stream(RefUtil.addRefs(networks)).forEach(ReferenceCountingBase::assertAlive);
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    RefUtil.freeRef(pipelineNetwork.add(new SumInputsLayer(), RefArrays.stream(RefUtil.addRefs(networks))
        .map(RefUtil.wrapInterface((Function<? super PipelineNetwork, ? extends InnerNode>) network -> {
          InnerNode temp_29_0001 = PipelineNetwork
              .transferNode(pipelineNetwork.addRef(), network.getHead());
          network.freeRef();
          return temp_29_0001;
        }, pipelineNetwork.addRef())).toArray(i -> new DAGNode[i])));
    RefUtil.freeRefs(networks);
    return pipelineNetwork;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SumInputsLayer[] addRefs(@Nullable SumInputsLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayer::addRef)
        .toArray((x) -> new SumInputsLayer[x]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_29_0008 = inObj[0].getData();
    @Nonnull final int[] dimensions = temp_29_0008.getDimensions();
    temp_29_0008.freeRef();
    if (3 != dimensions.length) {
      RefUtil.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList temp_29_0009 = inObj[i].getData();
      int[] dimensions1 = temp_29_0009.getDimensions();
      temp_29_0009.freeRef();
      if (Tensor.length(dimensions) != Tensor.length(dimensions1)) {
        TensorList temp_29_0010 = inObj[i].getData();
        IllegalArgumentException temp_29_0006 = new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(temp_29_0010.getDimensions()));
        temp_29_0010.freeRef();
        RefUtil.freeRefs(inObj);
        throw temp_29_0006;
      }
    }
    if (!CudaSystem.isEnabled()) {
      Layer temp_29_0011 = getCompatibilityLayer();
      Result temp_29_0007 = temp_29_0011.eval(RefUtil.addRefs(inObj));
      temp_29_0011.freeRef();
      RefUtil.freeRefs(inObj);
      return temp_29_0007;
    }
    RefStream<TensorList> tensorListStream = RefArrays.stream(RefUtil.addRefs(inObj)).map(x -> {
      TensorList temp_29_0002 = x.getData();
      x.freeRef();
      return temp_29_0002;
    });
    if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
      tensorListStream = tensorListStream.parallel();
    try {
      return new Result(RefUtil.get(tensorListStream.reduce((leftData, rightData) -> {
        TensorList temp_29_0003 = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, TensorList>) gpu -> {
              return gpu.addAndFree(precision, leftData.addRef(),
                  rightData == null ? null : rightData.addRef());
            }, rightData == null ? null : rightData.addRef(), leftData.addRef()),
            leftData.addRef(), rightData == null ? null : rightData.addRef());
        if (null != rightData)
          rightData.freeRef();
        leftData.freeRef();
        return temp_29_0003;
      })), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
          @Nonnull
          RefStream<Result> deltaStream = RefArrays.stream(RefUtil.addRefs(inObj));
          if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
            deltaStream = deltaStream.parallel();
          deltaStream.filter(Result::isAlive).forEach(RefUtil.wrapInterface((Consumer<? super Result>) obj -> {
            obj.accumulate(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
            obj.freeRef();
          }, buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef()));
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRefs(inObj);
        }
      }) {

        {
          RefUtil.addRefs(inObj);
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
          RefUtil.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRefs(inObj);
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
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumInputsLayer addRef() {
    return (SumInputsLayer) super.addRef();
  }
}
