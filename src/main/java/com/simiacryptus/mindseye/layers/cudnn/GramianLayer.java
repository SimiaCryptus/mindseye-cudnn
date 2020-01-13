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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

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
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public GramianLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static GramianLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GramianLayer(json, rs);
  }

  public static @SuppressWarnings("unused") GramianLayer[] addRefs(GramianLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GramianLayer::addRef).toArray((x) -> new GramianLayer[x]);
  }

  public static @SuppressWarnings("unused") GramianLayer[][] addRefs(GramianLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GramianLayer::addRefs)
        .toArray((x) -> new GramianLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inObj) {
    assert 1 == inObj.length;
    TensorList inputData = inObj[0].getData();
    int[] inputDimensions = inputData.getDimensions();
    assert 3 == inputDimensions.length;
    if (inputDimensions[0] == 1 && inputDimensions[1] == 1) {
      log.info("Suspicious Input: " + RefArrays.toString(inputDimensions));
    }
    final CudaTensorList tensorList = CudaSystem
        .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor tensor = gpu.getTensor(inputData == null ? null : inputData.addRef(), precision, MemoryType.Device,
              true);
          CudaTensorList temp_43_0002 = getOutput(gpu, tensor == null ? null : tensor.addRef());
          if (null != tensor)
            tensor.freeRef();
          return temp_43_0002;
        }, inputData == null ? null : inputData.addRef()), inputData == null ? null : inputData.addRef());
    try {
      try {
        try {
          return new Result(tensorList, new Result.Accumulator() {
            {
              Result.addRefs(inObj);
            }

            @Override
            public void accept(DeltaSet<UUID> buffer, TensorList delta) {
              @Nonnull
              final int[] outputDimensions = { 1, 1, inputDimensions[2] * inputDimensions[2] };
              if (!RefArrays.equals(delta.getDimensions(), outputDimensions)) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_43_0009 = new AssertionError(
                    RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
                if (null != delta)
                  delta.freeRef();
                throw temp_43_0009;
              }
              if (inObj[0].isAlive()) {
                final TensorList passbackTensorList = CudaSystem
                    .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                      @Nullable
                      final CudaTensor inputTensor = gpu.getTensor(inputData == null ? null : inputData.addRef(),
                          precision, MemoryType.Device, true);
                      CudaTensor deltaTensor = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                          MemoryType.Device, true);
                      CudaTensorList temp_43_0004 = GramianLayer.this.getFeedback(gpu,
                          inputTensor == null ? null : inputTensor.addRef(),
                          deltaTensor == null ? null : deltaTensor.addRef());
                      if (null != deltaTensor)
                        deltaTensor.freeRef();
                      if (null != inputTensor)
                        inputTensor.freeRef();
                      return temp_43_0004;
                    }, delta == null ? null : delta.addRef(), inputData == null ? null : inputData.addRef()),
                        delta == null ? null : delta.addRef());
                inObj[0].accumulate(buffer == null ? null : buffer.addRef(),
                    passbackTensorList == null ? null : passbackTensorList.addRef());
                if (null != passbackTensorList)
                  passbackTensorList.freeRef();
              }
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public @SuppressWarnings("unused") void _free() {
              if (null != inObj)
                ReferenceCounting.freeRefs(inObj);
            }
          }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                boolean temp_43_0005 = x.isAlive();
                if (null != x)
                  x.freeRef();
                return temp_43_0005;
              });
            }

            @Override
            public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
              Result.Accumulator temp_43_0010 = getAccumulator();
              temp_43_0010.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
              if (null != temp_43_0010)
                temp_43_0010.freeRef();
              if (null != delta)
                delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public void _free() {
              if (null != inObj)
                ReferenceCounting.freeRefs(inObj);
            }
          };
        } finally {
          if (null != inObj)
            ReferenceCounting.freeRefs(inObj);
        }
      } finally {
        if (null != tensorList)
          tensorList.freeRef();
      }
    } finally {
      if (null != inputData)
        inputData.freeRef();
    }

  }

  @Nonnull
  public CudaTensorList getFeedback(final CudnnHandle gpu, final CudaTensor inputTensor, final CudaTensor deltaTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    CudaMemory inputMemory = inputTensor.getMemory(gpu);
    CudaMemory deltaMemory = deltaTensor.getMemory(gpu);
    @Nonnull
    final int[] inputDimensions = { inputTensor.descriptor.width, inputTensor.descriptor.height,
        inputTensor.descriptor.channels };
    final int length = inputTensor.descriptor.batchCount;
    final int bands = inputDimensions[2];

    @Nullable
    final CudaMemory bufferMemory = gpu.allocate((long) inputTensor.descriptor.nStride * length * precision.size,
        MemoryType.Device, true);
    @Nonnull
    final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nonnull
    final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nullable
    final CudaMemory outputMemory = gpu.allocate((long) outputDescriptor.nStride * precision.size * length,
        MemoryType.Managed.ifEnabled(), true);
    @Nonnull
    final CudaMemory workspacePtr = gpu.allocate(Math.max(outputMemory.size, inputMemory.size), MemoryType.Device,
        true);
    @Nonnull
    final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);

    @Nonnull
    final CudaResource<cudnnOpTensorDescriptor> multiplyDescriptor = gpu
        .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
    CudaResource<cudnnReduceTensorDescriptor> reduceAddDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

    @Nonnull
    final CudaDevice.CudaTensorDescriptor bandDescriptor = gpu.newTensorDescriptor(precision, length, 1,
        inputDimensions[1], inputDimensions[0], inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
        inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
    @Nonnull
    final CudaDevice.CudaTensorDescriptor viewDescriptor1 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
        deltaTensor.descriptor.nStride, //
        deltaTensor.descriptor.cStride, //
        deltaTensor.descriptor.hStride, //
        deltaTensor.descriptor.wStride);
    @Nonnull
    final CudaDevice.CudaTensorDescriptor viewDescriptor2 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
        deltaTensor.descriptor.nStride, //
        deltaTensor.descriptor.cStride * bands, //
        deltaTensor.descriptor.hStride, //
        deltaTensor.descriptor.wStride //
    );

    if (null != deltaTensor)
      deltaTensor.freeRef();
    RefIntStream.range(0, bands).forEach(RefUtil.wrapInterface(band -> {
      CudaMemory deltaView1 = deltaMemory.withByteOffset(band * precision.size * bands);
      CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
          inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor1.getPtr(),
          deltaView1.getPtr(), precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      RefUtil.freeRef(inputMemory.dirty());
      RefUtil.freeRef(deltaView1.dirty());
      if (null != deltaView1)
        deltaView1.freeRef();
      RefUtil.freeRef(bufferMemory.dirty());
      CudaMemory deltaView2 = deltaMemory.withByteOffset(band * precision.size);
      CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
          inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor2.getPtr(),
          deltaView2.getPtr(), precision.getPointer(1.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      RefUtil.freeRef(inputMemory.dirty());
      RefUtil.freeRef(deltaView2.dirty());
      if (null != deltaView2)
        deltaView2.freeRef();
      RefUtil.freeRef(bufferMemory.dirty());
      CudaMemory outputViewMem = outputMemory.withByteOffset(bandDescriptor.cStride * band * precision.size);
      gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
          workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(), bufferMemory.getPtr(),
          precision.getPointer(0.0), bandDescriptor.getPtr(), outputViewMem.getPtr());
      RefUtil.freeRef(outputViewMem.dirty());
      if (null != outputViewMem)
        outputViewMem.freeRef();
      RefUtil.freeRef(bufferMemory.dirty());
    }, workspacePtr == null ? null : workspacePtr, bufferDescriptor == null ? null : bufferDescriptor,
        inputTensor == null ? null : inputTensor.addRef(), bufferMemory == null ? null : bufferMemory.addRef(),
        viewDescriptor1 == null ? null : viewDescriptor1, bandDescriptor == null ? null : bandDescriptor,
        viewDescriptor2 == null ? null : viewDescriptor2, inputMemory == null ? null : inputMemory.addRef(),
        deltaMemory == null ? null : deltaMemory.addRef(),
        reduceAddDescriptor == null ? null : reduceAddDescriptor.addRef(),
        multiplyDescriptor == null ? null : multiplyDescriptor, indexPtr == null ? null : indexPtr,
        outputMemory == null ? null : outputMemory.addRef()));

    if (null != inputTensor)
      inputTensor.freeRef();
    if (null != reduceAddDescriptor)
      reduceAddDescriptor.freeRef();
    if (null != bufferMemory)
      bufferMemory.freeRef();
    if (null != deltaMemory)
      deltaMemory.freeRef();
    if (null != inputMemory)
      inputMemory.freeRef();
    CudaTensorList temp_43_0006 = new CudaTensorList(new CudaTensor(outputMemory == null ? null : outputMemory.addRef(),
        outputDescriptor == null ? null : outputDescriptor, precision), length, inputDimensions, precision);
    if (null != outputMemory)
      outputMemory.freeRef();
    return temp_43_0006;
  }

  @Nonnull
  public CudaTensorList getOutput(final CudnnHandle gpu, final CudaTensor inputTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    @Nonnull
    final int[] inputDimensions = { inputTensor.descriptor.width, inputTensor.descriptor.height,
        inputTensor.descriptor.channels };
    final int length = inputTensor.descriptor.batchCount;
    final int bands = inputDimensions[2];
    @Nonnull
    final int[] outputDimensions = { 1, 1, bands * bands };

    CudaMemory inputMemory = inputTensor.getMemory(gpu);

    @Nonnull
    final CudaDevice.CudaTensorDescriptor ouputDescriptor = gpu.newTensorDescriptor(precision, length, bands * bands, 1,
        1, bands * bands, //
        1, //
        1, //
        1);
    @Nullable
    final CudaMemory outputMemory = gpu.allocate((long) ouputDescriptor.nStride * precision.size * length,
        MemoryType.Device, true);

    @Nonnull
    final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
        inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
        inputDimensions[0] * inputDimensions[1], //
        inputDimensions[0], //
        1);
    @Nullable
    final CudaMemory bufferMemory = gpu.allocate((long) bufferDescriptor.nStride * length * precision.size,
        MemoryType.Device, true);

    @Nonnull
    final CudaDevice.CudaTensorDescriptor inputViewDescriptor = gpu.newTensorDescriptor(precision, length, 1,
        inputDimensions[1], inputDimensions[0], inputTensor.descriptor.nStride, //
        inputTensor.descriptor.cStride, //
        inputTensor.descriptor.hStride, //
        inputTensor.descriptor.wStride);

    CudaResource<cudnnReduceTensorDescriptor> reduceAddDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

    @Nonnull
    final CudaDevice.CudaTensorDescriptor outputViewDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1,
        bands * bands, 1, 1, 1);
    @Nonnull
    final CudaResource<cudnnOpTensorDescriptor> multiplyDescriptor = gpu
        .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);

    @Nonnull
    final CudaMemory workspacePtr = gpu.allocate(Math.max(outputMemory.size, inputMemory.size), MemoryType.Device,
        true);
    @Nonnull
    final CudaMemory indexPtr = gpu.allocate((long) 12 * length, MemoryType.Device, true);
    RefIntStream.range(0, inputDimensions[2]).forEach(RefUtil.wrapInterface(band -> {
      CudaMemory inputView = inputMemory.withByteOffset(band * precision.size * inputTensor.descriptor.cStride);
      CudaSystem.handle(
          gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
              inputMemory.getPtr(), precision.getPointer(1.0), inputViewDescriptor.getPtr(), inputView.getPtr(),
              precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
      RefUtil.freeRef(bufferMemory.dirty());
      RefUtil.freeRef(inputView.dirty());
      if (null != inputView)
        inputView.freeRef();
      RefUtil.freeRef(inputMemory.dirty());
      CudaMemory outputView = outputMemory.withByteOffset(band * precision.size * bands);
      CudaSystem.handle(gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
          workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(),
          bufferMemory.getPtr(), precision.getPointer(0.0), outputViewDescriptor.getPtr(), outputView.getPtr()));
      RefUtil.freeRef(outputView.dirty());
      if (null != outputView)
        outputView.freeRef();
      RefUtil.freeRef(bufferMemory.dirty());
    }, outputMemory == null ? null : outputMemory.addRef(), bufferDescriptor == null ? null : bufferDescriptor,
        indexPtr == null ? null : indexPtr, inputViewDescriptor == null ? null : inputViewDescriptor,
        inputMemory == null ? null : inputMemory.addRef(),
        reduceAddDescriptor == null ? null : reduceAddDescriptor.addRef(),
        outputViewDescriptor == null ? null : outputViewDescriptor,
        multiplyDescriptor == null ? null : multiplyDescriptor, inputTensor == null ? null : inputTensor.addRef(),
        workspacePtr == null ? null : workspacePtr, bufferMemory == null ? null : bufferMemory.addRef()));

    if (null != inputTensor)
      inputTensor.freeRef();
    if (null != reduceAddDescriptor)
      reduceAddDescriptor.freeRef();
    RefUtil.freeRef(outputMemory.dirty());
    RefUtil.freeRef(bufferMemory.dirty());
    if (null != bufferMemory)
      bufferMemory.freeRef();
    RefUtil.freeRef(inputMemory.dirty());

    if (null != inputMemory)
      inputMemory.freeRef();
    CudaTensorList temp_43_0007 = new CudaTensorList(new CudaTensor(outputMemory == null ? null : outputMemory.addRef(),
        ouputDescriptor == null ? null : ouputDescriptor, precision), length, outputDimensions, precision);
    if (null != outputMemory)
      outputMemory.freeRef();
    return temp_43_0007;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("alpha", alpha);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") GramianLayer addRef() {
    return (GramianLayer) super.addRef();
  }
}
