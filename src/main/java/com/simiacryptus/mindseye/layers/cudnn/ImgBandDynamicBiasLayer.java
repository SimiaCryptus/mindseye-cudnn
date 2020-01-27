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
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

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

  @Nonnull
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
      Layer temp_33_0010 = getCompatibilityLayer();
      Result temp_33_0007 = temp_33_0010.eval(RefUtil.addRefs(inObj));
      temp_33_0010.freeRef();
      RefUtil.freeRefs(inObj);
      return temp_33_0007;
    }
    if (inObj.length != 2) {
      IllegalArgumentException temp_33_0008 = new IllegalArgumentException("inObj.length=" + inObj.length);
      RefUtil.freeRefs(inObj);
      throw temp_33_0008;
    }
    Result input = inObj[0].addRef();
    Result biasinput = inObj[1].addRef();
    TensorList biasData = biasinput.getData();
    if (1 != biasData.length()) {
      input.freeRef();
      biasinput.freeRef();
      IllegalArgumentException temp_33_0003 = new IllegalArgumentException("Input lengths: " + biasData.length());
      biasData.freeRef();
      RefUtil.freeRefs(inObj);
      throw temp_33_0003;
    }
    Tensor bias = biasData.get(0);
    biasData.freeRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != inputDimensions.length) {
      input.freeRef();
      biasinput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRefs(inObj);
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(inputDimensions));
    }
    if (0 == Tensor.length(inputData.getDimensions())) {
      biasinput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRefs(inObj);
      return input;
    }
    if (0 == bias.length()) {
      biasinput.freeRef();
      bias.freeRef();
      inputData.freeRef();
      RefUtil.freeRefs(inObj);
      return input;
    }
    try {
      //   assert !right.isAlive();
      return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
            @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
                .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
            @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                inputDimensions[2], inputDimensions[1], inputDimensions[0],
                inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
                inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision,
                MemoryType.Device, true);
            CudaMemory temp_33_0011 = gpu.allocate(bias.length() * precision.size, MemoryType.Device, true);
            CudaMemory biasMem = temp_33_0011.write(precision, bias.getData());
            temp_33_0011.freeRef();
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
            inputMemory.dirty();
            inputMemory.freeRef();
            biasMem.dirty();
            biasMem.freeRef();
            outputPtr.dirty();
            CudaTensor cudaTensor = new CudaTensor(outputPtr,
                outputDescriptor, precision);
            CudaTensorList temp_33_0006 = new CudaTensorList(cudaTensor.addRef(),
                length, inputDimensions, precision);
            cudaTensor.freeRef();
            return temp_33_0006;
          }, bias.addRef(), inputData.addRef()),
          inputData.addRef()), new Result.Accumulator() {
        {
          biasinput.addRef();
          bias.addRef();
          input.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
          if (biasinput.isAlive()) {
            @Nonnull
            double[] biasDelta = CudaSystem
                .run(RefUtil.wrapInterface((Function<CudnnHandle, double[]>) gpu -> {
                      @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(),
                          precision, MemoryType.Device, false);

                      CudaMemory temp_33_0012 = gpu.allocate(bias.length() * precision.size, MemoryType.Device,
                          true);
                      CudaMemory biasMem = temp_33_0012.write(precision, bias.getData());
                      temp_33_0012.freeRef();
                      int[] biasDim = bias.getDimensions();
                      CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1,
                          biasDim[2], biasDim[1], biasDim[0], biasDim[2] * biasDim[1] * biasDim[0],
                          biasDim[1] * biasDim[0], biasDim[0], 1);
                      CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                      assert deltaTensorMemory != null;
                      gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0),
                          deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                          precision.getPointer(0.0), biasDescriptor.getPtr(), biasMem.getPtr());
                      deltaTensorMemory.freeRef();
                      biasDescriptor.freeRef();
                      deltaTensor.freeRef();
                      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                      biasMem.dirty();
                      double[] biasV = new double[bias.length()];
                      biasMem.read(precision, biasV, 0);
                      biasMem.freeRef();
                      return biasV;
                    }, delta == null ? null : delta.addRef(), bias.addRef()),
                    delta == null ? null : delta.addRef());
            biasinput.accumulate(buffer == null ? null : buffer.addRef(),
                new TensorArray(new Tensor(biasDelta, bias.getDimensions())));
          }
          if (input.isAlive()) {
            input.accumulate(buffer == null ? null : buffer.addRef(),
                delta == null ? null : delta.addRef());
          }
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          biasinput.freeRef();
          bias.freeRef();
          input.freeRef();
        }
      }) {

        {
          RefUtil.addRefs(inObj);
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
          Accumulator temp_33_0013 = getAccumulator();
          assert temp_33_0013 != null;
          temp_33_0013.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
          temp_33_0013.freeRef();
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public void _free() {
          RefUtil.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRefs(inObj);
      inputData.freeRef();
      bias.freeRef();
      biasinput.freeRef();
      input.freeRef();
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

  public void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandDynamicBiasLayer addRef() {
    return (ImgBandDynamicBiasLayer) super.addRef();
  }
}
