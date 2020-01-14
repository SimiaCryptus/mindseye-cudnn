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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase implements MultiPrecision<ImgBandBiasLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  @Nullable
  private Tensor bias;

  public ImgBandBiasLayer(int bands) {
    this(new Tensor(1, 1, bands));
  }

  public ImgBandBiasLayer(@Nullable final Tensor bias) {
    Tensor temp_17_0001 = bias == null ? null : bias.addRef();
    if (null != this.bias)
      this.bias.freeRef();
    this.bias = temp_17_0001 == null ? null : temp_17_0001.addRef();
    if (null != temp_17_0001)
      temp_17_0001.freeRef();
    if (null != bias)
      bias.freeRef();
  }

  protected ImgBandBiasLayer(@Nonnull final JsonObject id, final Map<CharSequence, byte[]> rs) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    Tensor temp_17_0002 = Tensor.fromJson(id.get("bias"), rs);
    if (null != this.bias)
      this.bias.freeRef();
    this.bias = temp_17_0002 == null ? null : temp_17_0002.addRef();
    if (null != temp_17_0002)
      temp_17_0002.freeRef();
  }

  @Nonnull
  public double[] getBias() {
    assert bias != null;
    return bias.getData();
  }

  @Nonnull
  public ImgBandBiasLayer setBias(@Nullable Tensor bias) {
    Tensor temp_17_0003 = bias == null ? null : bias.addRef();
    if (null != this.bias)
      this.bias.freeRef();
    this.bias = temp_17_0003 == null ? null : temp_17_0003.addRef();
    if (null != temp_17_0003)
      temp_17_0003.freeRef();
    if (null != bias)
      bias.freeRef();
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
  public ImgBandBiasLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @Nonnull
  public ImgBandBiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    assert bias != null;
    RefUtil.freeRef(bias.setByCoord(c -> f.applyAsDouble(c.getIndex())));
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgBandBiasLayer[] addRefs(@Nullable ImgBandBiasLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRef)
        .toArray((x) -> new ImgBandBiasLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgBandBiasLayer[][] addRefs(@Nullable ImgBandBiasLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRefs)
        .toArray((x) -> new ImgBandBiasLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_17_0012 = getCompatibilityLayer();
      Result temp_17_0009 = temp_17_0012.eval(Result.addRefs(inObj));
      temp_17_0012.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_17_0009;
    }
    if (inObj.length != 1) {
      IllegalArgumentException temp_17_0010 = new IllegalArgumentException("inObj.length=" + inObj.length);
      ReferenceCounting.freeRefs(inObj);
      throw temp_17_0010;
    }
    Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(inputDimensions));
    }
    assert bias != null;
    if (bias.getDimensions()[2] != inputDimensions[2]) {
      input.freeRef();
      inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      throw new IllegalArgumentException(
          RefString.format("Input dimensions=%s; Bias dimensions=%s", RefArrays.toString(bias.getDimensions())));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return input;
    }
    if (0 == bias.length()) {
      inputData.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return input;
    }
    //   assert !right.isAlive();
    final ImgBandBiasLayer imgBandBiasLayer = ImgBandBiasLayer.this.addRef();
    try {
      try {
        try {
          try {
            return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
              try {
                @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                    .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
                @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                    inputDimensions[2], inputDimensions[1], inputDimensions[0],
                    inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
                    inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
                @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                    MemoryType.Device, true);
                CudaMemory temp_17_0013 = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true);
                CudaMemory biasMem = temp_17_0013.write(precision, bias.getData());
                temp_17_0013.freeRef();
                int[] biasDim = bias.getDimensions();
                CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2],
                    biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0],
                    1);
                //assert lPtr.size == rPtr.size;
                @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
                    MemoryType.Managed.ifEnabled(), true);
                CudaMemory inputMemory = inputTensor.getMemory(gpu);
                assert inputMemory != null;
                CudaSystem.handle(
                    gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
                        inputMemory.getPtr(), precision.getPointer(1.0), biasDescriptor.getPtr(), biasMem.getPtr(),
                        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
                biasDescriptor.freeRef();
                inputTensor.freeRef();
                opDescriptor.freeRef();
                assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                RefUtil.freeRef(inputMemory.dirty());
                inputMemory.freeRef();
                RefUtil.freeRef(biasMem.dirty());
                biasMem.freeRef();
                RefUtil.freeRef(outputPtr.dirty());
                CudaTensor cudaTensor = new CudaTensor(outputPtr,
                    outputDescriptor, precision);
                CudaTensorList temp_17_0008 = new CudaTensorList(cudaTensor.addRef(),
                    length, inputDimensions, precision);
                cudaTensor.freeRef();
                return temp_17_0008;
              } catch (Throwable e) {
                throw new RuntimeException(RefString.format("Error applying bias %s to input %s",
                    RefArrays.toString(bias.getDimensions()), RefArrays.toString(inputDimensions)), e);
              }
            }, inputData.addRef()), inputData.addRef()),
                new Result.Accumulator() {
                  {
                  }

                  @Override
                  public void accept(@Nonnull DeltaSet<UUID> buffer, @Nullable TensorList delta) {
                    if (!ImgBandBiasLayer.this.isFrozen()) {
                      @Nonnull
                      double[] biasDelta = CudaSystem
                          .run(RefUtil.wrapInterface((Function<CudnnHandle, double[]>) gpu -> {
                            @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                                precision, MemoryType.Device, false);

                            CudaMemory temp_17_0014 = gpu.allocate(bias.length() * precision.size, MemoryType.Device,
                                true);
                            CudaMemory biasMem = temp_17_0014.write(precision, bias.getData());
                            temp_17_0014.freeRef();
                            int[] biasDim = bias.getDimensions();
                            CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1,
                                biasDim[2], biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0],
                                biasDim[1] * biasDim[0], biasDim[0], 1);
                            CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                            assert deltaTensorMemory != null;
                            gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                                deltaTensorMemory.getPtr(), precision.getPointer(0.0), biasDescriptor.getPtr(),
                                biasMem.getPtr());
                            deltaTensorMemory.freeRef();
                            biasDescriptor.freeRef();
                            deltaTensor.freeRef();
                            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                            RefUtil.freeRef(biasMem.dirty());
                            double[] biasV = new double[bias.length()];
                            RefUtil.freeRef(biasMem.read(precision, biasV));
                            biasMem.freeRef();
                            return biasV;
                          }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
                      Delta<UUID> temp_17_0015 = buffer.get(imgBandBiasLayer.getId(),
                          bias == null ? null : bias.addRef());
                      assert temp_17_0015 != null;
                      RefUtil.freeRef(temp_17_0015.addInPlace(biasDelta));
                      temp_17_0015.freeRef();
                    }
                    if (input.isAlive()) {
                      input.accumulate(buffer.addRef(), delta == null ? null : delta.addRef());
                    }
                    if (null != delta)
                      delta.freeRef();
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
              public final void accumulate(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
                Result.Accumulator temp_17_0016 = getAccumulator();
                assert temp_17_0016 != null;
                temp_17_0016.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
                temp_17_0016.freeRef();
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
          imgBandBiasLayer.freeRef();
        }
      } finally {
        inputData.freeRef();
      }
    } finally {
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    assert bias != null;
    json.add("bias", bias.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    assert bias != null;
    return RefArrays.asList(bias.getData());
  }

  @Nonnull
  public ImgBandBiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
    return this.addRef();
  }

  @Nonnull
  public ImgBandBiasLayer set(@Nullable final Tensor tensor) {
    assert bias != null;
    bias.set(tensor == null ? null : tensor.addRef());
    if (null != tensor)
      tensor.freeRef();
    return this.addRef();
  }

  public void _free() {
    if (bias != null) {
      bias.freeRef();
      bias = null;
    }
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandBiasLayer addRef() {
    return (ImgBandBiasLayer) super.addRef();
  }
}
