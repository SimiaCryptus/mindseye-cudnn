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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public @RefAware
class SquareActivationLayer extends LayerBase implements MultiPrecision<SquareActivationLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private double alpha = 1.0;

  public SquareActivationLayer() {
  }

  protected SquareActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    this.alpha = id.getAsJsonPrimitive("alpha").getAsDouble();
  }

  public double getAlpha() {
    return alpha;
  }

  public SquareActivationLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this.addRef();
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
  public SquareActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static SquareActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SquareActivationLayer(json);
  }

  public static @SuppressWarnings("unused")
  SquareActivationLayer[] addRefs(SquareActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SquareActivationLayer::addRef)
        .toArray((x) -> new SquareActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SquareActivationLayer[][] addRefs(SquareActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SquareActivationLayer::addRefs)
        .toArray((x) -> new SquareActivationLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_38_0008 = getCompatibilityLayer();
      Result temp_38_0005 = temp_38_0008
          .eval(Result.addRefs(inObj));
      if (null != temp_38_0008)
        temp_38_0008.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_38_0005;
    }
    if (inObj.length != 1) {
      IllegalArgumentException temp_38_0006 = new IllegalArgumentException("inObj.length=" + inObj.length);
      if (null != inObj)
        ReferenceCounting.freeRefs(inObj);
      throw temp_38_0006;
    }
    Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != dimensions.length) {
      if (null != input)
        input.freeRef();
      if (null != inputData)
        inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    try {
      try {
        try {
          return new Result(CudaSystem.run(RefUtil.wrapInterface(
              (Function<CudnnHandle, CudaTensorList>) gpu -> {
                @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                    .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                    dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
                    dimensions[1] * dimensions[0], dimensions[0], 1);
                @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision,
                    MemoryType.Device, false);
                //assert inputTensor.size == rPtr.size;
                @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
                    MemoryType.Device, true);
                CudaMemory lPtrMemory = inputTensor.getMemory(gpu);
                CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(alpha),
                    inputTensor.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(1.0),
                    inputTensor.descriptor.getPtr(), lPtrMemory.getPtr(), precision.getPointer(0.0),
                    outputDescriptor.getPtr(), outputPtr.getPtr()));
                if (null != inputTensor)
                  inputTensor.freeRef();
                opDescriptor.freeRef();
                assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                RefUtil.freeRef(outputPtr.dirty());
                RefUtil.freeRef(lPtrMemory.dirty());
                if (null != lPtrMemory)
                  lPtrMemory.freeRef();
                RefUtil.freeRef(outputPtr.dirty());
                CudaTensor cudaTensor = new CudaTensor(outputPtr == null ? null : outputPtr,
                    outputDescriptor == null ? null : outputDescriptor, precision);
                CudaTensorList temp_38_0003 = new CudaTensorList(
                    cudaTensor == null ? null : cudaTensor.addRef(), length, dimensions, precision);
                if (null != cudaTensor)
                  cudaTensor.freeRef();
                return temp_38_0003;
              }, inputData == null ? null : inputData.addRef()), inputData == null ? null : inputData.addRef()),
              new Result.Accumulator() {
                {
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (input.isAlive()) {
                    @Nonnull
                    TensorList data = CudaSystem.run(RefUtil.wrapInterface(
                        (Function<CudnnHandle, CudaTensorList>) gpu -> {
                          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
                          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
                              length, dimensions[2], dimensions[1], dimensions[0],
                              dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                              dimensions[0], 1);
                          @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                              MemoryType.Device, true);
                          @Nullable final CudaTensor inputTensor = gpu.getTensor(input.getData(), precision, MemoryType.Device,
                              false);
                          //assert deltaTensor.size == inputTensor.size;
                          @Nonnull final CudaMemory outputPtr = gpu.allocate(
                              (long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
                          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                          CudaMemory rightTensorMemory = inputTensor.getMemory(gpu);
                          CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(2),
                              deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(), precision.getPointer(alpha),
                              inputTensor.descriptor.getPtr(), rightTensorMemory.getPtr(), precision.getPointer(0.0),
                              outputDescriptor.getPtr(), outputPtr.getPtr()));
                          if (null != inputTensor)
                            inputTensor.freeRef();
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          opDescriptor.freeRef();
                          RefUtil.freeRef(deltaTensorMemory.dirty());
                          if (null != deltaTensorMemory)
                            deltaTensorMemory.freeRef();
                          RefUtil.freeRef(rightTensorMemory.dirty());
                          if (null != rightTensorMemory)
                            rightTensorMemory.freeRef();
                          RefUtil.freeRef(outputPtr.dirty());
                          CudaTensor cudaTensor = new CudaTensor(outputPtr == null ? null : outputPtr,
                              outputDescriptor == null ? null : outputDescriptor, precision);
                          CudaTensorList temp_38_0004 = new CudaTensorList(
                              cudaTensor == null ? null : cudaTensor.addRef(), length, dimensions, precision);
                          if (null != cudaTensor)
                            cudaTensor.freeRef();
                          return temp_38_0004;
                        }, delta == null ? null : delta.addRef(), input == null ? null : input.addRef()),
                        delta == null ? null : delta.addRef());
                    input.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
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
            public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
              Result.Accumulator temp_38_0009 = getAccumulator();
              temp_38_0009.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
              if (null != temp_38_0009)
                temp_38_0009.freeRef();
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
      } finally {
        if (null != inputData)
          inputData.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("alpha", alpha);
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
  SquareActivationLayer addRef() {
    return (SquareActivationLayer) super.addRef();
  }
}
