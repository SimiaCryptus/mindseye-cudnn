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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
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
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class NProductLayer extends LayerBase implements MultiPrecision<NProductLayer> {

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
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static NProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new NProductLayer(json);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  NProductLayer[] addRefs(@Nullable NProductLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NProductLayer::addRef)
        .toArray((x) -> new NProductLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  NProductLayer[][] addRefs(@Nullable NProductLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NProductLayer::addRefs)
        .toArray((x) -> new NProductLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_36_0010 = getCompatibilityLayer();
      Result temp_36_0008 = temp_36_0010.eval(Result.addRefs(inObj));
      temp_36_0010.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_36_0008;
    }
    if (inObj.length <= 1) {
      IllegalArgumentException temp_36_0009 = new IllegalArgumentException("inObj.length=" + inObj.length);
      ReferenceCounting.freeRefs(inObj);
      throw temp_36_0009;
    }
    TensorList temp_36_0011 = inObj[0].getData();
    @Nonnull final int[] dimensions = temp_36_0011.getDimensions();
    temp_36_0011.freeRef();
    TensorList temp_36_0012 = inObj[0].getData();
    final int length = temp_36_0012.length();
    temp_36_0012.freeRef();
    if (3 != dimensions.length) {
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList data = inObj[i].getData();
      if (Tensor.length(dimensions) != Tensor.length(data.getDimensions())) {
        IllegalArgumentException temp_36_0001 = new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(data.getDimensions()));
        data.freeRef();
        ReferenceCounting.freeRefs(inObj);
        throw temp_36_0001;
      }
      data.freeRef();
    }
    try {
      return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, TensorList>) gpu -> {
        @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
            .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
            dimensions[1] * dimensions[0], dimensions[0], 1);
        TensorList temp_36_0002 = RefUtil.get(RefArrays.stream(Result.addRefs(inObj)).map(x -> {
          TensorList temp_36_0003 = x.getData();
          x.freeRef();
          return temp_36_0003;
        }).reduce(RefUtil.wrapInterface((BinaryOperator<TensorList>) (l, r) -> {
          @Nullable final CudaTensor lPtr = gpu.getTensor(l == null ? null : l.addRef(), precision, MemoryType.Device, false);
          if (null != l)
            l.freeRef();
          @Nullable final CudaTensor rPtr = gpu.getTensor(r == null ? null : r.addRef(), precision, MemoryType.Device, false);
          if (null != r)
            r.freeRef();
          @Nonnull final CudaMemory outputPtr = gpu.allocate((long) outputDescriptor.nStride * length * precision.size,
              MemoryType.Device, true);
          CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
          CudaMemory rPtrMemory = rPtr.getMemory(gpu.addRef());
          assert rPtrMemory != null;
          assert lPtrMemory != null;
          CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(), precision.getPointer(1.0),
              lPtr.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0), rPtr.descriptor.getPtr(),
              rPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
          rPtr.freeRef();
          lPtr.freeRef();
          RefUtil.freeRef(lPtrMemory.dirty());
          lPtrMemory.freeRef();
          RefUtil.freeRef(rPtrMemory.dirty());
          rPtrMemory.freeRef();
          RefUtil.freeRef(outputPtr.dirty());
          CudaTensorList temp_36_0004 = new CudaTensorList(
              new CudaTensor(outputPtr.addRef(),
                  outputDescriptor.addRef(), precision),
              length, dimensions, precision);
          outputPtr.freeRef();
          return temp_36_0004;
        }, opDescriptor, outputDescriptor)));
        return temp_36_0002;
      }, Result.addRefs(inObj)), RefArrays.stream(Result.addRefs(inObj)).map(Result::getData).toArray()),
          new Result.Accumulator() {
            {
              Result.addRefs(inObj);
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
              for (int index = 0; index < inObj.length; index++) {
                final Result input = inObj[index].addRef();
                if (input.isAlive()) {
                  final int _index = index;
                  @Nonnull
                  TensorList data = RefUtil.get(RefIntStream.range(0, inObj.length)
                      .mapToObj(RefUtil.wrapInterface((IntFunction<TensorList>) i -> {
                        return i == _index ? delta.addRef() : inObj[i].getData();
                      }, Result.addRefs(inObj), delta.addRef())).reduce((l, r) -> {
                        CudaTensorList temp_36_0005 = CudaSystem
                            .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                                  @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                                      .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                                  @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(
                                      precision, length, dimensions[2], dimensions[1], dimensions[0],
                                      dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                                      dimensions[0], 1);
                                  @Nullable final CudaTensor lPtr = gpu.getTensor(l.addRef(), precision,
                                      MemoryType.Device, false);
                                  @Nullable final CudaTensor rPtr = gpu.getTensor(r == null ? null : r.addRef(), precision,
                                      MemoryType.Device, false);
                                  @Nonnull final CudaMemory outputPtr = gpu.allocate(
                                      (long) outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
                                  CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                                  CudaMemory rPtrMemory = rPtr.getMemory(gpu);
                                  assert rPtrMemory != null;
                                  assert lPtrMemory != null;
                                  CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(),
                                      precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
                                      precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
                                      precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
                                  rPtr.freeRef();
                                  lPtr.freeRef();
                                  opDescriptor.freeRef();
                                  RefUtil.freeRef(lPtrMemory.dirty());
                                  lPtrMemory.freeRef();
                                  RefUtil.freeRef(rPtrMemory.dirty());
                                  rPtrMemory.freeRef();
                                  RefUtil.freeRef(outputPtr.dirty());
                                  CudaTensorList temp_36_0006 = new CudaTensorList(
                                      new CudaTensor(outputPtr.addRef(),
                                          outputDescriptor.addRef(), precision),
                                      length, dimensions, precision);
                                  outputPtr.freeRef();
                                  outputDescriptor.freeRef();
                                  return temp_36_0006;
                                }, l.addRef(), r == null ? null : r.addRef()),
                                l.addRef(), r == null ? null : r.addRef());
                        if (null != r)
                          r.freeRef();
                        l.freeRef();
                        return temp_36_0005;
                      }));
                  input.accumulate(buffer == null ? null : buffer.addRef(), data);
                }
                input.freeRef();
              }
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

        @Override
        public final void accumulate(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
          Result.Accumulator temp_36_0013 = getAccumulator();
          assert temp_36_0013 != null;
          temp_36_0013.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          temp_36_0013.freeRef();
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  NProductLayer addRef() {
    return (NProductLayer) super.addRef();
  }
}
