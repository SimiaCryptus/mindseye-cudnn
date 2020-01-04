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
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgBandBiasLayer extends LayerBase
    implements MultiPrecision<ImgBandBiasLayer> {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private Tensor bias;

  public ImgBandBiasLayer(int bands) {
    this(new Tensor(1, 1, bands));
  }

  public ImgBandBiasLayer(final Tensor bias) {
    this.bias = bias;
  }

  protected ImgBandBiasLayer(@Nonnull final JsonObject id,
      final com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    this.bias = Tensor.fromJson(id.get("bias"), rs);
  }

  public double[] getBias() {
    return bias.getData();
  }

  public ImgBandBiasLayer setBias(Tensor bias) {
    this.bias = bias;
    return this;
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
    return this;
  }

  @Nonnull
  public ImgBandBiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    bias.setByCoord(c -> f.applyAsDouble(c.getIndex()));
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    if (inObj.length != 1) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull
    final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      throw new IllegalArgumentException(
          "dimensions=" + com.simiacryptus.ref.wrappers.RefArrays.toString(inputDimensions));
    }
    if (bias.getDimensions()[2] != inputDimensions[2]) {
      throw new IllegalArgumentException(String.format("Input dimensions=%s; Bias dimensions=%s",
          com.simiacryptus.ref.wrappers.RefArrays.toString(bias.getDimensions())));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      return input;
    }
    if (0 == bias.length()) {
      return input;
    }
    //   assert !right.isAlive();
    return new Result(CudaSystem.run(gpu -> {
      try {
        @Nonnull
        final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
            .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
        @Nonnull
        final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            inputDimensions[2], inputDimensions[1], inputDimensions[0],
            inputDimensions[2] * inputDimensions[1] * inputDimensions[0], inputDimensions[1] * inputDimensions[0],
            inputDimensions[0], 1);
        @Nullable
        final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
        CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true).write(precision,
            bias.getData());
        int[] biasDim = bias.getDimensions();
        CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2], biasDim[1],
            biasDim[0], biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0], 1);
        //assert lPtr.size == rPtr.size;
        @Nonnull
        final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
            MemoryType.Managed.ifEnabled(), true);
        CudaMemory inputMemory = inputTensor.getMemory(gpu);
        CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0),
            inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), biasDescriptor.getPtr(),
            biasMem.getPtr(), precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputMemory.dirty();
        biasMem.dirty();
        outputPtr.dirty();
        CudaTensor cudaTensor = new CudaTensor(outputPtr, outputDescriptor, precision);
        return new CudaTensorList(cudaTensor, length, inputDimensions, precision);
      } catch (Throwable e) {
        throw new RuntimeException(String.format("Error applying bias %s to input %s",
            com.simiacryptus.ref.wrappers.RefArrays.toString(bias.getDimensions()),
            com.simiacryptus.ref.wrappers.RefArrays.toString(inputDimensions)), e);
      }
    }, inputData), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        @Nonnull
        double[] biasDelta = CudaSystem.run(gpu -> {
          @Nullable
          final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);

          CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true).write(precision,
              bias.getData());
          int[] biasDim = bias.getDimensions();
          CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2], biasDim[1],
              biasDim[0], biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0], 1);
          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
          gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
              deltaTensorMemory.getPtr(), precision.getPointer(0.0), biasDescriptor.getPtr(), biasMem.getPtr());
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          biasMem.dirty();
          double[] biasV = new double[bias.length()];
          biasMem.read(precision, biasV);
          return biasV;
        }, delta);
        buffer.get(ImgBandBiasLayer.this.getId(), bias).addInPlace(biasDelta);
      }
      if (input.isAlive()) {
        input.accumulate(buffer, delta);
      }
    }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull
        final Result element : inObj)
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
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.add("bias", bias.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(bias.getData());
  }

  @Nonnull
  public ImgBandBiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
    return this;
  }

  public ImgBandBiasLayer set(final Tensor tensor) {
    bias.set(tensor);
    return this;
  }

  public void _free() {
    if (this.bias != null) {
      bias = null;
    }
    super._free();
  }

  public @Override @SuppressWarnings("unused") ImgBandBiasLayer addRef() {
    return (ImgBandBiasLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[] addRefs(ImgBandBiasLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRef)
        .toArray((x) -> new ImgBandBiasLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[][] addRefs(ImgBandBiasLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRefs)
        .toArray((x) -> new ImgBandBiasLayer[x][]);
  }
}
