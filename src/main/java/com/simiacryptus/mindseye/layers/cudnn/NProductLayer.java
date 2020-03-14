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
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.BinaryOperator;
import java.util.function.IntFunction;

import static com.simiacryptus.mindseye.lang.Result.getData;

@SuppressWarnings("serial")
public class NProductLayer extends LayerBase implements MultiPrecision {

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

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static NProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new NProductLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    final int inLength = inObj.length;
    if (inLength <= 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("inObj.length=" + inLength);
    }
    TensorList data0 = inObj[0].getData();
    @Nonnull final int[] dimensions = data0.getDimensions();
    final int length = data0.length();
    data0.freeRef();
    if (3 != dimensions.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inLength; i++) {
      TensorList data = inObj[i].getData();
      int[] dataDimensions = data.getDimensions();
      data.freeRef();
      if (Tensor.length(dimensions) != Tensor.length(dataDimensions)) {
        RefUtil.freeRef(inObj);
        throw new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(dataDimensions));
      }
    }
    boolean alive = alive(RefUtil.addRef(inObj));
    Accumulator accumulator = new Accumulator(precision, inLength, length, dimensions, RefUtil.addRef(inObj));
    TensorList data = fwd(dimensions, length, inObj);
    return new Result(data, accumulator, alive);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  NProductLayer addRef() {
    return (NProductLayer) super.addRef();
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  @NotNull
  private TensorList fwd(int[] dimensions, int length, @Nonnull Result[] inObj) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, TensorList>) gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
          dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
          dimensions[1] * dimensions[0], dimensions[0], 1);
      return RefUtil.get(RefArrays.stream(RefUtil.addRef(inObj)).map(x -> {
        return getData(x);
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
        lPtrMemory.dirty();
        lPtrMemory.freeRef();
        rPtrMemory.dirty();
        rPtrMemory.freeRef();
        outputPtr.dirty();
        return new CudaTensorList(
            new CudaTensor(outputPtr,
                outputDescriptor.addRef(), precision),
            length, dimensions, precision);
      }, opDescriptor, outputDescriptor, gpu)));
    }, RefUtil.addRef(inObj)), RefArrays.stream(inObj).map(result -> {
      TensorList data = result.getData();
      result.freeRef();
      return data;
    }).toArray());
  }

  private static class Accumulator extends Result.Accumulator {

    private final int inLength;
    private final int length;
    private final int[] dimensions;
    private final Result[] inObj;
    private Precision precision;

    public Accumulator(Precision precision, int inLength, int length, int[] dimensions, Result... inObj) {
      this.inLength = inLength;
      this.length = length;
      this.dimensions = dimensions;
      this.inObj = inObj;
      this.precision = precision;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      for (int index = 0; index < inLength; index++) {
        final Result input = inObj[index].addRef();
        if (input.isAlive()) {
          final int _index = index;
          @Nonnull
          TensorList data = RefUtil.get(RefIntStream.range(0, inLength)
              .mapToObj(RefUtil.wrapInterface((IntFunction<TensorList>) i -> {
                return i == _index ? delta.addRef() : inObj[i].getData();
              }, RefUtil.addRef(inObj), delta.addRef())).reduce((l, r) -> {
                return CudaSystem
                    .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(
                              precision, length, dimensions[2], dimensions[1], dimensions[0],
                              dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                              dimensions[0], 1);
                          @Nullable final CudaTensor lPtr = gpu.getTensor(l.addRef(), precision,
                              MemoryType.Device, false);
                          @Nullable final CudaTensor rPtr = gpu.getTensor(r == null ? null : r.addRef(), precision,
                              MemoryType.Device, false);
                          @Nonnull final CudaMemory outputPtr = gpu.allocate(
                              (long) outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
                          CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
                          CudaMemory rPtrMemory = rPtr.getMemory(gpu.addRef());
                          assert rPtrMemory != null;
                          assert lPtrMemory != null;
                          CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(),
                              precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
                              precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
                              precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
                          gpu.freeRef();
                          rPtr.freeRef();
                          lPtr.freeRef();
                          opDescriptor.freeRef();
                          lPtrMemory.dirty();
                          lPtrMemory.freeRef();
                          rPtrMemory.dirty();
                          rPtrMemory.freeRef();
                          outputPtr.dirty();
                          return new CudaTensorList(
                              new CudaTensor(outputPtr, outputDescriptor, precision),
                              length, dimensions, precision);
                        }, l.addRef(), r == null ? null : r.addRef()
                    ), l, r);
              }));
          DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          Result.Accumulator accumulator = input.getAccumulator();
          try {
            accumulator.accept(buffer1, data);
          } finally {
            accumulator.freeRef();
          }
        }
        input.freeRef();
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
