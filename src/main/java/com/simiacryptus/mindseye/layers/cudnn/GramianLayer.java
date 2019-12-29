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
import jcuda.jcudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class GramianLayer extends LayerBase implements MultiPrecision<GramianLayer> {
  private static final Logger log = LoggerFactory.getLogger(GramianLayer.class);

  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private double alpha = 1.0;

  public GramianLayer() {
  }

  public GramianLayer(UUID id) {
    super(id, "Gramian");
  }

  protected GramianLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    this.alpha = json.getAsJsonPrimitive("alpha").getAsDouble();
  }

  public double getAlpha() {
    return alpha;
  }

  public GramianLayer setAlpha(final double alpha) {
    this.alpha = alpha;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public GramianLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @SuppressWarnings("unused")
  public static GramianLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GramianLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    assert 1 == inObj.length;
    TensorList inputData = inObj[0].getData();
    int[] inputDimensions = inputData.getDimensions();
    assert 3 == inputDimensions.length;
    if (inputDimensions[0] == 1 && inputDimensions[1] == 1) {
      log.info("Suspicious Input: " + Arrays.toString(inputDimensions));
    }
    final CudaTensorList tensorList = CudaSystem.run(gpu -> {
      CudaTensor tensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
      return getOutput(gpu, tensor);
    }, inputData);
    return new Result(tensorList, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      @Nonnull final int[] outputDimensions = {1, 1, inputDimensions[2] * inputDimensions[2]};
      if (!Arrays.equals(delta.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      if (inObj[0].isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
          CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
          return getFeedback(gpu, inputTensor, deltaTensor);
        }, delta);
        inObj[0].accumulate(buffer, passbackTensorList);
      }
    }) {

      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
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
  public CudaTensorList getFeedback(final CudnnHandle gpu, final CudaTensor inputTensor, final CudaTensor deltaTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    CudaMemory inputMemory = inputTensor.getMemory(gpu);
    CudaMemory deltaMemory = deltaTensor.getMemory(gpu);
    @Nonnull final int[] inputDimensions = {inputTensor.descriptor.width, inputTensor.descriptor.height,
        inputTensor.descriptor.channels};
    final int length = inputTensor.descriptor.batchCount;
    final int bands = inputDimensions[2];

    @Nullable final CudaMemory bufferMemory = gpu.allocate((long) inputTensor.descriptor.nStride * length * precision.size,
        MemoryType.Device, true);
    @Nonnull final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nullable final CudaMemory outputMemory = gpu.allocate((long) outputDescriptor.nStride * precision.size * length,
        MemoryType.Managed.ifEnabled(), true);
    @Nonnull final CudaMemory workspacePtr = gpu.allocate(Math.max(outputMemory.size, inputMemory.size), MemoryType.Device,
        true);
    @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

    @Nonnull final CudaResource<cudnnOpTensorDescriptor> multiplyDescriptor = gpu
        .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
    CudaResource<cudnnReduceTensorDescriptor> reduceAddDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

    @Nonnull final CudaDevice.CudaTensorDescriptor bandDescriptor = gpu.newTensorDescriptor(precision, length, 1,
        inputDimensions[1], inputDimensions[0], inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
        inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
    @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor1 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
        deltaTensor.descriptor.nStride, //
        deltaTensor.descriptor.cStride, //
        deltaTensor.descriptor.hStride, //
        deltaTensor.descriptor.wStride);
    @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor2 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
        deltaTensor.descriptor.nStride, //
        deltaTensor.descriptor.cStride * bands, //
        deltaTensor.descriptor.hStride, //
        deltaTensor.descriptor.wStride //
    );

    IntStream.range(0, bands).forEach(band -> {
      CudaMemory deltaView1 = deltaMemory.withByteOffset(band * precision.size * bands);
      CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
          inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor1.getPtr(),
          deltaView1.getPtr(), precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      inputMemory.dirty();
      deltaView1.dirty();
      bufferMemory.dirty();
      CudaMemory deltaView2 = deltaMemory.withByteOffset(band * precision.size);
      CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
          inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor2.getPtr(),
          deltaView2.getPtr(), precision.getPointer(1.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      inputMemory.dirty();
      deltaView2.dirty();
      bufferMemory.dirty();
      CudaMemory outputViewMem = outputMemory.withByteOffset(bandDescriptor.cStride * band * precision.size);
      gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(), bufferMemory.getPtr(),
          precision.getPointer(0.0), bandDescriptor.getPtr(), outputViewMem.getPtr());
      outputViewMem.dirty();
      bufferMemory.dirty();
    });

    return new CudaTensorList(new CudaTensor(outputMemory, outputDescriptor, precision), length, inputDimensions, precision);
  }

  @Nonnull
  public CudaTensorList getOutput(final CudnnHandle gpu, final CudaTensor inputTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    @Nonnull final int[] inputDimensions = {inputTensor.descriptor.width, inputTensor.descriptor.height,
        inputTensor.descriptor.channels};
    final int length = inputTensor.descriptor.batchCount;
    final int bands = inputDimensions[2];
    @Nonnull final int[] outputDimensions = {1, 1, bands * bands};

    CudaMemory inputMemory = inputTensor.getMemory(gpu);

    @Nonnull final CudaDevice.CudaTensorDescriptor ouputDescriptor = gpu.newTensorDescriptor(precision, length, bands * bands, 1,
        1, bands * bands, //
        1, //
        1, //
        1);
    @Nullable final CudaMemory outputMemory = gpu.allocate((long) ouputDescriptor.nStride * precision.size * length,
        MemoryType.Device, true);

    @Nonnull final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nullable final CudaMemory bufferMemory = gpu.allocate((long) bufferDescriptor.nStride * length * precision.size,
        MemoryType.Device, true);

    @Nonnull final CudaDevice.CudaTensorDescriptor inputViewDescriptor = gpu.newTensorDescriptor(precision, length, 1,
        inputDimensions[1], inputDimensions[0], inputTensor.descriptor.nStride, //
        inputTensor.descriptor.cStride, //
        inputTensor.descriptor.hStride, //
        inputTensor.descriptor.wStride);

    CudaResource<cudnnReduceTensorDescriptor> reduceAddDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

    @Nonnull final CudaDevice.CudaTensorDescriptor outputViewDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1,
        bands * bands, 1, 1, 1);
    @Nonnull final CudaResource<cudnnOpTensorDescriptor> multiplyDescriptor = gpu
        .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);

    @Nonnull final CudaMemory workspacePtr = gpu.allocate(Math.max(outputMemory.size, inputMemory.size), MemoryType.Device,
        true);
    @Nonnull final CudaMemory indexPtr = gpu.allocate((long) 12 * length, MemoryType.Device, true);
    IntStream.range(0, inputDimensions[2]).forEach(band -> {
      CudaMemory inputView = inputMemory.withByteOffset(band * precision.size * inputTensor.descriptor.cStride);
      CudaSystem.handle(
          gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
              inputMemory.getPtr(), precision.getPointer(1.0), inputViewDescriptor.getPtr(), inputView.getPtr(),
              precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      bufferMemory.dirty();
      inputView.dirty();
      inputMemory.dirty();
      CudaMemory outputView = outputMemory.withByteOffset(band * precision.size * bands);
      CudaSystem.handle(gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
          workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(),
          bufferMemory.getPtr(), precision.getPointer(0.0), outputViewDescriptor.getPtr(), outputView.getPtr()));
      outputView.dirty();
      bufferMemory.dirty();
    });

    outputMemory.dirty();
    bufferMemory.dirty();
    inputMemory.dirty();

    return new CudaTensorList(new CudaTensor(outputMemory, ouputDescriptor, precision), length, outputDimensions, precision);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("alpha", alpha);
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
