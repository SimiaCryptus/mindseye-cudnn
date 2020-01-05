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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class ValueLayer extends LayerBase {

  private final Precision precision;
  private final CudaTensorList tensorList;

  protected ValueLayer(@Nonnull final JsonObject json,
                       Map<CharSequence, byte[]> resources) {
    super(json);
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    Tensor value = Tensor.fromJson(json.get("value"), resources);
    this.tensorList = toDevice(value, precision);
  }

  public ValueLayer(final Tensor data) {
    super();
    this.precision = Precision.Float;
    this.tensorList = toDevice(data, precision);
    this.frozen = true;
  }

  @SuppressWarnings("unused")
  public static ValueLayer fromJson(@Nonnull final JsonObject json,
                                    Map<CharSequence, byte[]> rs) {
    return new ValueLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ValueLayer[] addRefs(ValueLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ValueLayer::addRef)
        .toArray((x) -> new ValueLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ValueLayer[][] addRefs(ValueLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ValueLayer::addRefs)
        .toArray((x) -> new ValueLayer[x][]);
  }

  public CudaTensorList toDevice(final Tensor data, final Precision precision) {
    if (null == data)
      return null;
    return CudaSystem.run(gpu -> {
      CudaMemory cudaMemory = gpu.allocate(data.length() * precision.size, MemoryType.Managed.ifEnabled(), true);
      cudaMemory.write(precision, data.getData());
      int[] dimensions = data.getDimensions();
      CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, 1, dimensions[2],
          dimensions[1], dimensions[0]);
      return new CudaTensorList(new CudaTensor(cudaMemory, tensorDescriptor, precision), 1, dimensions, precision);
    });
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... array) {
    assert 0 == array.length;
    return new Result(tensorList, new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList data) {
      }
    }) {

      @Override
      public boolean isAlive() {
        return false;
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    Tensor tensor = tensorList.get(0);
    json.add("value", tensor.getJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor tensor = tensorList.get(0);
    return RefArrays.asList(tensor.getData());
  }

  public void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ValueLayer addRef() {
    return (ValueLayer) super.addRef();
  }
}
