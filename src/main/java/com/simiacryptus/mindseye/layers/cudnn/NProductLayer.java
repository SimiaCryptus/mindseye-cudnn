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
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class NProductLayer extends LayerBase
    implements MultiPrecision<NProductLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public NProductLayer() {
  }

  protected NProductLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(ProductInputsLayer.class);
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public NProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static NProductLayer fromJson(@Nonnull final JsonObject json,
                                       Map<CharSequence, byte[]> rs) {
    return new NProductLayer(json);
  }

  public static @SuppressWarnings("unused")
  NProductLayer[] addRefs(NProductLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NProductLayer::addRef)
        .toArray((x) -> new NProductLayer[x]);
  }

  public static @SuppressWarnings("unused")
  NProductLayer[][] addRefs(NProductLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NProductLayer::addRefs)
        .toArray((x) -> new NProductLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    if (inObj.length <= 1) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    @Nonnull final int[] dimensions = inObj[0].getData().getDimensions();
    final int length = inObj[0].getData().length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList data = inObj[i].getData();
      if (Tensor.length(dimensions) != Tensor.length(data.getDimensions())) {
        throw new IllegalArgumentException(RefArrays.toString(dimensions) + " != "
            + RefArrays.toString(data.getDimensions()));
      }
    }
    return new Result(CudaSystem.run(gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, dimensions[2],
          dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
          dimensions[0], 1);
      @Nonnull final TensorList result1 = RefArrays.stream(inObj).map(x -> {
        return x.getData();
      }).reduce((l, r) -> {
        @Nullable final CudaTensor lPtr = gpu.getTensor(l, precision, MemoryType.Device, false);
        @Nullable final CudaTensor rPtr = gpu.getTensor(r, precision, MemoryType.Device, false);
        //assert lPtr.memory.size == rPtr.memory.size;
        @Nonnull final CudaMemory outputPtr = gpu.allocate((long) outputDescriptor.nStride * length * precision.size,
            MemoryType.Device, true);
        CudaMemory lPtrMemory = lPtr.getMemory(gpu);
        CudaMemory rPtrMemory = rPtr.getMemory(gpu);
        CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(), precision.getPointer(1.0),
            lPtr.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0), rPtr.descriptor.getPtr(),
            rPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
        lPtrMemory.dirty();
        rPtrMemory.dirty();
        outputPtr.dirty();
        return new CudaTensorList(new CudaTensor(outputPtr, outputDescriptor, precision), length, dimensions,
            precision);
      }).get();
      return result1;
    }, RefArrays.stream(inObj).map(Result::getData).toArray()),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          for (int index = 0; index < inObj.length; index++) {
            final Result input = inObj[index];
            if (input.isAlive()) {
              final int _index = index;
              @Nonnull
              TensorList data = RefIntStream.range(0, inObj.length).mapToObj(i -> {
                return i == _index ? delta : inObj[i].getData();
              }).reduce((l, r) -> {
                return CudaSystem.run(gpu -> {
                  @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                      .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                  @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                      dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
                      dimensions[1] * dimensions[0], dimensions[0], 1);

                  @Nullable final CudaTensor lPtr = gpu.getTensor(l, precision, MemoryType.Device, false);
                  @Nullable final CudaTensor rPtr = gpu.getTensor(r, precision, MemoryType.Device, false);
                  //assert lPtr.memory.size == rPtr.memory.size;
                  @Nonnull final CudaMemory outputPtr = gpu.allocate((long) outputDescriptor.nStride * length * precision.size,
                      MemoryType.Device, true);
                  CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                  CudaMemory rPtrMemory = rPtr.getMemory(gpu);
                  CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(), precision.getPointer(1.0),
                      lPtr.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0),
                      rPtr.descriptor.getPtr(), rPtrMemory.getPtr(), precision.getPointer(0.0),
                      outputDescriptor.getPtr(), outputPtr.getPtr()));
                  lPtrMemory.dirty();
                  rPtrMemory.dirty();
                  outputPtr.dirty();
                  return new CudaTensorList(new CudaTensor(outputPtr, outputDescriptor, precision), length, dimensions,
                      precision);
                }, l, r);
              }).get();
              input.accumulate(buffer, data);
            }
          }
        }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }

      @Override
      public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      public void _free() {
        for (int i = 0; i < inObj.length; i++) {
          inObj[i].getData();
        }
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
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
  NProductLayer addRef() {
    return (NProductLayer) super.addRef();
  }
}
