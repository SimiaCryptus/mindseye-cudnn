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
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ImgBandDynamicBiasLayer extends LayerBase implements MultiPrecision {

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  public ImgBandDynamicBiasLayer() {
  }

  protected ImgBandDynamicBiasLayer(@Nonnull final JsonObject id, final Map<CharSequence, byte[]> rs) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
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
  public static ImgBandDynamicBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandDynamicBiasLayer(json, rs);
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
    if (inLength != 2) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("inObj.length=" + inLength);
    }
    Result input = inObj[0].addRef();
    Result biasInput = inObj[1].addRef();
    TensorList biasData = biasInput.getData();
    int biasLength = biasData.length();
    if (1 != biasLength) {
      input.freeRef();
      biasInput.freeRef();
      biasData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("Input lengths: " + biasLength);
    }
    Tensor bias = biasData.get(0);
    biasData.freeRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      biasInput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(inputDimensions));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      biasInput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    if (0 == bias.length()) {
      biasInput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRef(inObj);
      return input;
    }
    //   assert !right.isAlive();
    CudaTensorList data = fwd(bias.addRef(), inputData, inputDimensions, length);
    Accumulator accumulator = new Accumulator(bias, biasInput.getAccumulator(), biasInput.isAlive(), input.getAccumulator(), input.isAlive());
    biasInput.freeRef();
    input.freeRef();
    boolean alive = alive(inObj);
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

  public void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandDynamicBiasLayer addRef() {
    return (ImgBandDynamicBiasLayer) super.addRef();
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  @NotNull
  private CudaTensorList fwd(Tensor bias, TensorList inputData, int[] inputDimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
          inputDimensions[2], inputDimensions[1], inputDimensions[0],
          inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
          inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
          MemoryType.Device, true);
      CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true);
      biasMem.write(precision, bias.addRef());
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
    }, bias, inputData.addRef()), inputData);
  }

  private class Accumulator extends Result.Accumulator {

    private final Tensor bias;
    private Result.Accumulator biasinputAccumulator;
    private Result.Accumulator inputAccumulator;
    private boolean inputAlive;
    private boolean alive;

    public Accumulator(Tensor bias, Result.Accumulator biasinputAccumulator, boolean biasinputAlive, Result.Accumulator inputAccumulator, boolean inputAlive) {
      this.bias = bias;
      this.biasinputAccumulator = biasinputAccumulator;
      this.inputAccumulator = inputAccumulator;
      this.inputAlive = inputAlive;
      alive = biasinputAlive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      if (alive) {
        @Nonnull
        Tensor biasDelta = CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, Tensor>) gpu -> {
                  @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                      precision, MemoryType.Device, false);

                  CudaMemory temp_33_0012 = gpu.allocate(bias.length() * precision.size, MemoryType.Device,
                      true);
                  temp_33_0012.write(precision, bias.addRef());
                  CudaMemory biasMem = temp_33_0012.addRef();
                  temp_33_0012.freeRef();
                  int[] biasDim = bias.getDimensions();
                  CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1,
                      biasDim[2], biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0],
                      biasDim[1] * biasDim[0], biasDim[0], 1);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
                  assert deltaTensorMemory != null;
                  gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                      precision.getPointer(0.0), biasDescriptor.getPtr(), biasMem.getPtr());
                  deltaTensorMemory.freeRef();
                  biasDescriptor.freeRef();
                  deltaTensor.freeRef();
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  gpu.freeRef();
                  biasMem.dirty();
                  Tensor biasV = new Tensor(bias.getDimensions());
                  biasMem.read(precision, biasV.addRef(), 0);
                  biasMem.freeRef();
                  return biasV;
                }, delta == null ? null : delta.addRef(), bias.addRef()),
                delta == null ? null : delta.addRef());
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        biasinputAccumulator.accept(buffer1, new TensorArray(biasDelta));
      }
      if (inputAlive) {
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        TensorList delta1 = delta == null ? null : delta.addRef();
        inputAccumulator.accept(buffer1, delta1);
      }
      if (null != delta)
        delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      biasinputAccumulator.freeRef();
      bias.freeRef();
      inputAccumulator.freeRef();
    }
  }
}
