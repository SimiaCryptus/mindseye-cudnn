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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  @Nullable
  private Tensor bias;

  public ImgBandBiasLayer(int bands) {
    this(new Tensor(1, 1, bands));
  }

  public ImgBandBiasLayer(@Nullable final Tensor bias) {
    setBias(bias);
  }

  protected ImgBandBiasLayer(@Nonnull final JsonObject id, final Map<CharSequence, byte[]> rs) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    if (null != this.bias)
      this.bias.freeRef();
    this.bias = Tensor.fromJson(id.get("bias"), rs);
  }

  @Nonnull
  public double[] getBias() {
    assert bias != null;
    return bias.getData();
  }

  public void setBias(@Nullable Tensor bias) {
    if (null != this.bias)
      this.bias.freeRef();
    this.bias = bias;
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
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  public void setWeights(@Nonnull IntToDoubleFunction f) {
    assert bias != null;
    bias.setByCoord(c -> f.applyAsDouble(c.getIndex()));
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json, rs);
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
    int inLength = inObj.length;
    if (inLength != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("inObj.length=" + inLength);
    }
    Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(inputDimensions));
    }
    assert bias != null;
    if (bias.getDimensions()[2] != inputDimensions[2]) {
      input.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException(
          RefString.format("Input dimensions=%s; Bias dimensions=%s", RefArrays.toString(bias.getDimensions())));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    if (0 == bias.length()) {
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    //   assert !right.isAlive();
    CudaTensorList data = fwd(inputData, inputDimensions, length);
    boolean alive = Result.anyAlive(inObj);
    Accumulator accumulator = new Accumulator(this.getId(), bias.addRef(), precision, isFrozen(), input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(data, accumulator, alive || !isFrozen());
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

  public void addWeights(@Nonnull DoubleSupplier f) {
    Util.add(f, getBias());
  }

  public void set(@Nullable Tensor tensor) {
    assert bias != null;
    bias.set(tensor);
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

  @NotNull
  private CudaTensorList fwd(TensorList inputData, int[] inputDimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      try {
        @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
            .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            inputDimensions[2], inputDimensions[1], inputDimensions[0],
            inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
            inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
        @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
            MemoryType.Device, true);
        CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true);
        biasMem.write(precision, bias.getData());
        int[] biasDim = bias.getDimensions();
        CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2],
            biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0],
            1);
        //assert lPtr.size == rPtr.size;
        @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
            MemoryType.Managed.ifEnabled(), true);
        CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());
        assert inputMemory != null;
        CudaSystem.handle(
            gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
                inputMemory.getPtr(), precision.getPointer(1.0), biasDescriptor.getPtr(), biasMem.getPtr(),
                precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
        biasDescriptor.freeRef();
        inputTensor.freeRef();
        opDescriptor.freeRef();
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        gpu.freeRef();
        inputMemory.dirty();
        inputMemory.freeRef();
        biasMem.dirty();
        biasMem.freeRef();
        outputPtr.dirty();
        return new CudaTensorList(
            new CudaTensor(outputPtr, outputDescriptor, precision),
            length, inputDimensions, precision);
      } catch (Throwable e) {
        throw new RuntimeException(RefString.format("Error applying bias %s to input %s",
            RefArrays.toString(bias.getDimensions()), RefArrays.toString(inputDimensions)), e);
      }
    }, inputData.addRef()), inputData);
  }

  private static class Accumulator extends Result.Accumulator {

    private UUID id;
    private Tensor bias;
    private Precision precision;
    private boolean frozen;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(UUID id, Tensor bias, Precision precision, boolean frozen, Result.Accumulator accumulator, boolean alive) {
      this.id = id;
      this.bias = bias;
      this.precision = precision;
      this.frozen = frozen;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (!frozen) {
        @Nonnull
        double[] biasDelta = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, double[]>) gpu -> {
              @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                  precision, MemoryType.Device, false);

              CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device,
                  true);
              biasMem.write(precision, bias.getData());
              int[] biasDim = bias.getDimensions();
              CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1,
                  biasDim[2], biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0],
                  biasDim[1] * biasDim[0], biasDim[0], 1);
              CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
              assert deltaTensorMemory != null;
              gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                  deltaTensorMemory.getPtr(), precision.getPointer(0.0), biasDescriptor.getPtr(),
                  biasMem.getPtr());
              deltaTensorMemory.freeRef();
              biasDescriptor.freeRef();
              deltaTensor.freeRef();
              assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
              biasMem.dirty();
              double[] biasV = new double[bias.length()];
              biasMem.read(precision, biasV, 0);
              biasMem.freeRef();
              gpu.freeRef();
              return biasV;
            }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
        Delta<UUID> temp_17_0015 = buffer.get(id, bias == null ? null : bias.addRef());
        assert temp_17_0015 != null;
        temp_17_0015.addInPlace(biasDelta);
        temp_17_0015.freeRef();
      }
      if (alive) {
        this.accumulator.accept(buffer, delta);
      } else {
        if (null != delta)
          delta.freeRef();
        buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      bias.freeRef();
      if(null != accumulator) accumulator.freeRef();
    }
  }
}
