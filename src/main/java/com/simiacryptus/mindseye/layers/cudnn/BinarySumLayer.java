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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.Util;
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
class BinarySumLayer extends LayerBase implements MultiPrecision<BinarySumLayer> {

  private double leftFactor;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private double rightFactor;

  public BinarySumLayer() {
    this(1.0, 1.0);
  }

  public BinarySumLayer(final double leftFactor, final double rightFactor) {
    this.leftFactor = leftFactor;
    this.rightFactor = rightFactor;
  }

  protected BinarySumLayer(@Nonnull final JsonObject json) {
    super(json);
    rightFactor = json.get("rightFactor").getAsDouble();
    leftFactor = json.get("leftFactor").getAsDouble();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    @Nonnull
    PipelineNetwork network = new PipelineNetwork(2);
    LinearActivationLayer temp_51_0012 = new LinearActivationLayer();
    LinearActivationLayer temp_51_0013 = new LinearActivationLayer();
    LinearActivationLayer temp_51_0014 = temp_51_0012.setScale(this.leftFactor);
    LinearActivationLayer temp_51_0015 = temp_51_0013.setScale(this.rightFactor);
    RefUtil
        .freeRef(network.add(new SumInputsLayer(), network.add(temp_51_0014.freeze(), network.getInput(0)),
            network.add(temp_51_0015.freeze(), network.getInput(1))));
    if (null != temp_51_0015)
      temp_51_0015.freeRef();
    if (null != temp_51_0014)
      temp_51_0014.freeRef();
    if (null != temp_51_0013)
      temp_51_0013.freeRef();
    if (null != temp_51_0012)
      temp_51_0012.freeRef();
    return network;

  }

  public double getLeftFactor() {
    return leftFactor;
  }

  @Nonnull
  public void setLeftFactor(final double leftFactor) {
    this.leftFactor = leftFactor;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public BinarySumLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public double getRightFactor() {
    return rightFactor;
  }

  @Nonnull
  public void setRightFactor(final double rightFactor) {
    this.rightFactor = rightFactor;
  }

  @SuppressWarnings("unused")
  public static BinarySumLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BinarySumLayer(json);
  }

  public static @SuppressWarnings("unused")
  BinarySumLayer[] addRefs(BinarySumLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinarySumLayer::addRef)
        .toArray((x) -> new BinarySumLayer[x]);
  }

  public static @SuppressWarnings("unused")
  BinarySumLayer[][] addRefs(BinarySumLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinarySumLayer::addRefs)
        .toArray((x) -> new BinarySumLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length == 1) {
      if (rightFactor != 1) {
        ReferenceCounting.freeRefs(inObj);
        throw new IllegalStateException();
      }
      if (leftFactor != 1) {
        ReferenceCounting.freeRefs(inObj);
        throw new IllegalStateException();
      }
      Result temp_51_0007 = inObj[0];
      ReferenceCounting.freeRefs(inObj);
      return temp_51_0007;
    }
    if (inObj.length > 2) {
      if (rightFactor != 1) {
        ReferenceCounting.freeRefs(inObj);
        throw new IllegalStateException();
      }
      if (leftFactor != 1) {
        ReferenceCounting.freeRefs(inObj);
        throw new IllegalStateException();
      }
      Result temp_51_0008 = RefArrays
          .stream(Result.addRefs(inObj)).reduce((a, b) -> {
            Result temp_51_0001 = eval(a == null ? null : a.addRef(),
                b == null ? null : b.addRef());
            if (null != b)
              b.freeRef();
            if (null != a)
              a.freeRef();
            return temp_51_0001;
          }).get();
      ReferenceCounting.freeRefs(inObj);
      return temp_51_0008;
    }
    assert (inObj.length == 2);
    final TensorList leftData = inObj[0].getData();
    final TensorList rightData = inObj[1].getData();
    int[] leftDimensions = leftData.getDimensions();
    if (3 < leftDimensions.length) {
      if (null != leftData)
        leftData.freeRef();
      if (null != rightData)
        rightData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    @Nonnull final int[] dimensions = {leftDimensions.length < 1 ? 0 : leftDimensions[0],
        leftDimensions.length < 2 ? 1 : leftDimensions[1], leftDimensions.length < 3 ? 1 : leftDimensions[2]};
    final int length = leftData.length();
    if (length != rightData.length()) {
      if (null != leftData)
        leftData.freeRef();
      if (null != rightData)
        rightData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException();
    }
    if (3 != dimensions.length) {
      if (null != leftData)
        leftData.freeRef();
      if (null != rightData)
        rightData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      TensorList temp_51_0016 = inObj[i].getData();
      if (Tensor.length(dimensions) != Tensor.length(temp_51_0016.getDimensions())) {
        if (null != leftData)
          leftData.freeRef();
        if (null != rightData)
          rightData.freeRef();
        TensorList temp_51_0017 = inObj[i].getData();
        IllegalArgumentException temp_51_0009 = new IllegalArgumentException(
            RefArrays.toString(dimensions) + " != " + RefArrays.toString(temp_51_0017.getDimensions()));
        if (null != temp_51_0017)
          temp_51_0017.freeRef();
        if (null != inObj)
          ReferenceCounting.freeRefs(inObj);
        throw temp_51_0009;
      }
      if (null != temp_51_0016)
        temp_51_0016.freeRef();
    }
    if (!CudaSystem.isEnabled()) {
      if (null != leftData)
        leftData.freeRef();
      if (null != rightData)
        rightData.freeRef();
      Layer temp_51_0018 = getCompatibilityLayer();
      Result temp_51_0010 = temp_51_0018
          .eval(Result.addRefs(inObj));
      if (null != temp_51_0018)
        temp_51_0018.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_51_0010;
    }
    try {
      try {
        try {
          return new Result(CudaSystem.run(RefUtil.wrapInterface(
              (Function<CudnnHandle, CudaTensorList>) gpu -> {
                @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                    .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
                @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                    dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
                    dimensions[1] * dimensions[0], dimensions[0], 1);
                @Nullable final CudaTensor lPtr = gpu.getTensor(leftData == null ? null : leftData.addRef(), precision,
                    MemoryType.Device, false);
                @Nullable final CudaTensor rPtr = gpu.getTensor(rightData == null ? null : rightData.addRef(), precision,
                    MemoryType.Device, false);
                @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * Tensor.length(dimensions) * length,
                    MemoryType.Managed.ifEnabled(), true);
                CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                CudaMemory rPtrMemory = rPtr.getMemory(gpu);
                gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(leftFactor), lPtr.descriptor.getPtr(),
                    lPtrMemory.getPtr(), precision.getPointer(rightFactor), rPtr.descriptor.getPtr(),
                    rPtrMemory.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
                if (null != rPtr)
                  rPtr.freeRef();
                if (null != lPtr)
                  lPtr.freeRef();
                opDescriptor.freeRef();
                assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                RefUtil.freeRef(lPtrMemory.dirty());
                if (null != lPtrMemory)
                  lPtrMemory.freeRef();
                RefUtil.freeRef(rPtrMemory.dirty());
                if (null != rPtrMemory)
                  rPtrMemory.freeRef();
                RefUtil.freeRef(outputPtr.dirty());
                CudaTensor cudaTensor = new CudaTensor(outputPtr == null ? null : outputPtr,
                    outputDescriptor == null ? null : outputDescriptor, precision);
                CudaTensorList temp_51_0004 = new CudaTensorList(
                    cudaTensor == null ? null : cudaTensor.addRef(), length, dimensions, precision);
                if (null != cudaTensor)
                  cudaTensor.freeRef();
                return temp_51_0004;
              }, rightData == null ? null : rightData.addRef(), leftData == null ? null : leftData.addRef()),
              leftData == null ? null : leftData.addRef()), new Result.Accumulator() {
            {
              Result.addRefs(inObj);
            }

            @Override
            public void accept(DeltaSet<UUID> buffer, TensorList delta) {

              Runnable a = RefUtil.wrapInterface(() -> {
                    if (inObj[0].isAlive()) {
                      CudaTensorList tensorList = CudaSystem.run(RefUtil.wrapInterface(
                          (Function<CudnnHandle, CudaTensorList>) gpu -> {
                            @Nullable final CudaTensor lPtr = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                                MemoryType.Device, false);
                            @Nonnull final CudaMemory passbackPtr = gpu.allocate(
                                precision.size * Tensor.length(dimensions) * length, MemoryType.Managed.ifEnabled(),
                                true);
                            @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                                precision, length, dimensions[2], dimensions[1], dimensions[0],
                                dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                                dimensions[0], 1);
                            CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                            gpu.cudnnTransformTensor(precision.getPointer(leftFactor), lPtr.descriptor.getPtr(),
                                lPtrMemory.getPtr(), precision.getPointer(0.0), passbackDescriptor.getPtr(),
                                passbackPtr.getPtr());
                            if (null != lPtr)
                              lPtr.freeRef();
                            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                            RefUtil.freeRef(passbackPtr.dirty());
                            RefUtil.freeRef(lPtrMemory.dirty());
                            if (null != lPtrMemory)
                              lPtrMemory.freeRef();
                            CudaTensor cudaTensor = new CudaTensor(passbackPtr == null ? null : passbackPtr,
                                passbackDescriptor == null ? null : passbackDescriptor, precision);
                            CudaTensorList temp_51_0005 = new CudaTensorList(
                                cudaTensor == null ? null : cudaTensor.addRef(), length, dimensions, precision);
                            if (null != cudaTensor)
                              cudaTensor.freeRef();
                            return temp_51_0005;
                          }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
                      inObj[0].accumulate(buffer == null ? null : buffer.addRef(),
                          tensorList == null ? null : tensorList.addRef());
                      if (null != tensorList)
                        tensorList.freeRef();
                    }
                  }, delta == null ? null : delta.addRef(), buffer == null ? null : buffer.addRef(),
                  Result.addRefs(inObj));
              Runnable b = RefUtil.wrapInterface(() -> {
                    if (inObj[1].isAlive()) {
                      CudaTensorList tensorList = CudaSystem.run(RefUtil.wrapInterface(
                          (Function<CudnnHandle, CudaTensorList>) gpu -> {
                            @Nullable final CudaTensor lPtr = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                                MemoryType.Device, false);
                            @Nonnull final CudaMemory outputPtr = gpu.allocate(
                                precision.size * Tensor.length(dimensions) * length, MemoryType.Managed.ifEnabled(),
                                true);
                            @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                                precision, length, dimensions[2], dimensions[1], dimensions[0],
                                dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                                dimensions[0], 1);
                            CudaMemory lPtrMemory = lPtr.getMemory(gpu);
                            gpu.cudnnTransformTensor(precision.getPointer(rightFactor), lPtr.descriptor.getPtr(),
                                lPtrMemory.getPtr(), precision.getPointer(0.0), passbackDescriptor.getPtr(),
                                outputPtr.getPtr());
                            if (null != lPtr)
                              lPtr.freeRef();
                            RefUtil.freeRef(outputPtr.dirty());
                            RefUtil.freeRef(lPtrMemory.dirty());
                            if (null != lPtrMemory)
                              lPtrMemory.freeRef();
                            CudaTensorList temp_51_0006 = new CudaTensorList(
                                new CudaTensor(outputPtr == null ? null : outputPtr,
                                    passbackDescriptor == null ? null : passbackDescriptor, precision),
                                length, dimensions, precision);
                            return temp_51_0006;
                          }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
                      inObj[1].accumulate(buffer == null ? null : buffer.addRef(),
                          tensorList == null ? null : tensorList.addRef());
                      if (null != tensorList)
                        tensorList.freeRef();
                    }
                  }, buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef(),
                  Result.addRefs(inObj));
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
              if (CoreSettings.INSTANCE().isSingleThreaded())
                Util.runAllSerial(a, b);
              else
                Util.runAllParallel(a, b);
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
      } finally {
        if (null != rightData)
          rightData.freeRef();
      }
    } finally {
      if (null != leftData)
        leftData.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("rightFactor", rightFactor);
    json.addProperty("leftFactor", leftFactor);
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
  BinarySumLayer addRef() {
    return (BinarySumLayer) super.addRef();
  }
}
