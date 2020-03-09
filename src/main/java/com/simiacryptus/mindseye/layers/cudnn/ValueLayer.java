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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ValueLayer extends LayerBase {

  private final Precision precision;
  @Nonnull
  private final CudaTensorList tensorList;

  protected ValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    this.tensorList = toDevice(Tensor.fromJson(json.get("value"), resources), precision);
  }

  public ValueLayer(@Nullable final Tensor data) {
    super();
    this.precision = Precision.Float;
    this.tensorList = toDevice(data, precision);
    this.frozen = true;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ValueLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ValueLayer(json, rs);
  }


  @Nullable
  public CudaTensorList toDevice(@Nullable final Tensor data, @Nonnull final Precision precision) {
    if (null == data) {
      return null;
    }
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaMemory cudaMemory = gpu.allocate(data.length() * precision.size, MemoryType.Managed.ifEnabled(), true);
      cudaMemory.write(precision, data.addRef());
      int[] dimensions = data.getDimensions();
      CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, 1, dimensions[2],
          dimensions[1], dimensions[0]);
      gpu.freeRef();
      return new CudaTensorList(
          new CudaTensor(cudaMemory, tensorDescriptor, precision),
          1, dimensions, precision);
    }, data));
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... array) {
    assert 0 == array.length;
    RefUtil.freeRef(array);
    return new Result(tensorList.addRef(), new Accumulator());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    Tensor tensor = tensorList.get(0);
    json.add("value", tensor.getJson(resources, dataSerializer));
    tensor.freeRef();
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor tensor = tensorList.get(0);
    RefList<double[]> refList = RefArrays.asList(tensor.getData());
    tensor.freeRef();
    return refList;
  }

  public void _free() {
    super._free();
    tensorList.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ValueLayer addRef() {
    return (ValueLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {
    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList data) {
      if (null != data)
        data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }
  }
}
