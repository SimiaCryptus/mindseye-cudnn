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
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ImgBandDynamicBiasLayer extends LayerBase implements MultiPrecision<ImgBandDynamicBiasLayer> {

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

  @Nonnull
  @Override
  public ImgBandDynamicBiasLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgBandDynamicBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandDynamicBiasLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    if (inObj.length != 2) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result input = inObj[0];
    Result biasinput = inObj[1];
    TensorList biasData = biasinput.getData();
    if (1 != biasData.length()) {
      throw new IllegalArgumentException("Input lengths: " + biasData.length());
    }
    Tensor bias = biasData.get(0);
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(inputDimensions));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      return input;
    }
    if (0 == bias.length()) {
      return input;
    }
    //   assert !right.isAlive();
    return new Result(CudaSystem.run(gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
          .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
          inputDimensions[2], inputDimensions[1], inputDimensions[0],
          inputDimensions[2] * inputDimensions[1] * inputDimensions[0], inputDimensions[1] * inputDimensions[0],
          inputDimensions[0], 1);
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
      CudaMemory biasMem = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true).write(precision,
          bias.getData());
      int[] biasDim = bias.getDimensions();
      CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2], biasDim[1],
          biasDim[0], biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0], 1);
      //assert lPtr.size == rPtr.size;
      @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length,
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
    }, inputData), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (biasinput.isAlive()) {
        @Nonnull
        double[] biasDelta = CudaSystem.run(gpu -> {
          @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);

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
        biasinput.accumulate(buffer, new TensorArray(new Tensor(biasDelta, bias.getDimensions())));
      }
      if (input.isAlive()) {
        input.accumulate(buffer, delta);
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

      @Override
      protected void _free() {
      }

    };
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
  public List<double[]> state() {
    return Arrays.asList();
  }

  @Override
  protected void _free() {
    super._free();
  }
}
