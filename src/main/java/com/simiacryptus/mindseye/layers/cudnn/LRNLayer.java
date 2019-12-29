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
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnLRNMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class LRNLayer extends LayerBase implements MultiPrecision<LRNLayer> {
  private static final Logger log = LoggerFactory.getLogger(LRNLayer.class);

  private int width;
  private double alpha;
  private double beta;
  private double k;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;

  private LRNLayer() {
  }

  public LRNLayer(int width) {
    this(width, 1e-4, 0.75, 2.0);
  }

  public LRNLayer(int width, double alpha, double beta, double k) {
    this.setWidth(width);
    this.setAlpha(alpha);
    this.setBeta(beta);
    this.setK(k);
  }

  protected LRNLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    setWidth(json.get("width").getAsInt());
    setAlpha(json.get("alpha").getAsDouble());
    setBeta(json.get("beta").getAsDouble());
    setK(json.get("k").getAsDouble());
    JsonPrimitive precision = json.getAsJsonPrimitive("precision");
    if (null != precision) {
      this.setPrecision(Precision.valueOf(precision.getAsString()));
    } else {
      this.setPrecision(CudaSettings.INSTANCE().defaultPrecision);
    }
    assert 0 < getWidth();
    assert 0 < getAlpha();
  }

  public double getAlpha() {
    return alpha;
  }

  public LRNLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public double getBeta() {
    return beta;
  }

  public LRNLayer setBeta(double beta) {
    this.beta = beta;
    return this;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return null;
  }

  public double getK() {
    return k;
  }

  public LRNLayer setK(double k) {
    this.k = k;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public LRNLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  public int getWidth() {
    return width;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  @SuppressWarnings("unused")
  public static LRNLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LRNLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled())
      return getCompatibilityLayer().eval(inObj);
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    @Nonnull final int[] outputSize = new int[]{length, inputSize[2], inputSize[1], inputSize[0]};
    final CudaTensor outputData = CudaSystem.run(gpu -> {
      try {
        gpu.initThread();
        @Nonnull final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(this.getWidth(), this.getAlpha(),
            this.getBeta(), this.getK());
        @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, getPrecision(), MemoryType.Device, false);
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(getPrecision(), outputSize[0],
            outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3],
            outputSize[2] * outputSize[3], outputSize[3], 1);
        @Nonnull final CudaMemory outputTensor = gpu.allocate((long) getPrecision().size * Tensor.length(outputSize),
            MemoryType.Managed.ifEnabled(), true);
        CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
        CudaSystem
            .handle(gpu.cudnnLRNCrossChannelForward(descriptor.getPtr(), cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1,
                getPrecision().getPointer(1.0), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                getPrecision().getPointer(0.0), outputDescriptor.getPtr(), outputTensor.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputDataMemory.dirty();
        outputTensor.dirty();
        return new CudaTensor(outputTensor, outputDescriptor, getPrecision());
      } catch (@Nonnull final Throwable e) {
        throw new ComponentException("Error", e);
      }
    }, inputData);
    return new Result(new CudaTensorList(outputData, length, new int[]{outputSize[3], outputSize[2], outputSize[1]}, getPrecision()),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
          assert error.length() == inputData.length();
          if (input.isAlive()) {
            TensorList data = CudaSystem.run(gpu -> {
              @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(getPrecision(), length,
                  inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0],
                  inputSize[1] * inputSize[0], inputSize[0], 1);
              @Nonnull final CudaResource<cudnnLRNDescriptor> descriptor = gpu.createLRNDescriptor(this.getWidth(),
                  this.getAlpha(), this.getBeta(), this.getK());
              @Nullable final CudaTensor inputTensor;
              synchronized (gpu) {
                inputTensor = gpu.getTensor(inputData, getPrecision(), MemoryType.Device, true);
              }
              @Nullable final CudaTensor errorPtr;
              synchronized (gpu) {
                errorPtr = gpu.getTensor(error, getPrecision(), MemoryType.Device, true);
              }
              @Nonnull final CudaMemory passbackBuffer = gpu.allocate((long) inputDims * getPrecision().size * length,
                  MemoryType.Managed.ifEnabled(), true);
              CudaMemory outputDataMemory = outputData.getMemory(gpu);
              CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
              CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
              CudaSystem.handle(gpu.cudnnLRNCrossChannelBackward(descriptor.getPtr(),
                  cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1, getPrecision().getPointer(1.0),
                  outputData.descriptor.getPtr(), outputDataMemory.getPtr(), errorPtr.descriptor.getPtr(),
                  errorPtrMemory.getPtr(), inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                  getPrecision().getPointer(0.0), passbackDescriptor.getPtr(), passbackBuffer.getPtr()));
              outputDataMemory.dirty();
              errorPtrMemory.dirty();
              inputDataMemory.dirty();
              passbackBuffer.dirty();
              return new CudaTensorList(new CudaTensor(passbackBuffer, passbackDescriptor, getPrecision()), length, inputSize, getPrecision());
            }, error);
            input.accumulate(buffer, data);
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      @Override
      protected void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", getAlpha());
    json.addProperty("beta", getBeta());
    json.addProperty("k", getK());
    json.addProperty("width", getWidth());
    json.addProperty("precision", getPrecision().name());
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
