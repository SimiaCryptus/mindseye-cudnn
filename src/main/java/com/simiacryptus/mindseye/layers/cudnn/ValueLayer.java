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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ValueLayer extends LayerBase {

  private final Precision precision;
  @Nonnull
  private final CudaTensorList tensorList;

  protected ValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    Tensor value = Tensor.fromJson(json.get("value"), resources);
    CudaTensorList temp_19_0001 = toDevice(value == null ? null : value.addRef(), precision);
    this.tensorList = temp_19_0001 == null ? null : temp_19_0001.addRef();
    if (null != temp_19_0001)
      temp_19_0001.freeRef();
    if (null != value)
      value.freeRef();
  }

  public ValueLayer(@Nullable final Tensor data) {
    super();
    this.precision = Precision.Float;
    CudaTensorList temp_19_0002 = toDevice(data == null ? null : data.addRef(), precision);
    this.tensorList = temp_19_0002 == null ? null : temp_19_0002.addRef();
    if (null != temp_19_0002)
      temp_19_0002.freeRef();
    if (null != data)
      data.freeRef();
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
    CudaTensorList temp_19_0005 = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
      CudaMemory cudaMemory = gpu.allocate(data.length() * precision.size, MemoryType.Managed.ifEnabled(), true);
      RefUtil.freeRef(cudaMemory.write(precision, data.getData()));
      int[] dimensions = data.getDimensions();
      CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, 1, dimensions[2],
          dimensions[1], dimensions[0]);
      CudaTensorList temp_19_0003 = new CudaTensorList(new CudaTensor(cudaMemory.addRef(),
          tensorDescriptor.addRef(), precision), 1, dimensions, precision);
      tensorDescriptor.freeRef();
      cudaMemory.freeRef();
      return temp_19_0003;
    }, data.addRef()));
    data.freeRef();
    return temp_19_0005;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... array) {
    assert 0 == array.length;
    ReferenceCounting.freeRefs(array);
    return new Result(tensorList.addRef(), new Result.Accumulator() {
      @Override
      public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList data) {
        if (null != data)
          data.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }

      public @SuppressWarnings("unused")
      void _free() {
      }
    });
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
    RefList<double[]> temp_19_0004 = RefArrays.asList(tensor.getData());
    tensor.freeRef();
    return temp_19_0004;
  }

  public void _free() {
    tensorList.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ValueLayer addRef() {
    return (ValueLayer) super.addRef();
  }
}
