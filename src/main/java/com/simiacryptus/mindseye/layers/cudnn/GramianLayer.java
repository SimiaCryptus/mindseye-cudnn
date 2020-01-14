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

  @Nonnull
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

  @Nonnull
  @SuppressWarnings("unused")
  public static GramianLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GramianLayer(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  GramianLayer[] addRefs(@Nullable GramianLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GramianLayer::addRef).toArray((x) -> new GramianLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  GramianLayer[][] addRefs(@Nullable GramianLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GramianLayer::addRefs)
        .toArray((x) -> new GramianLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorList inputData = inObj[0].getData();
    int[] inputDimensions = inputData.getDimensions();
    assert 3 == inputDimensions.length;
    if (inputDimensions[0] == 1 && inputDimensions[1] == 1) {
      log.info("Suspicious Input: " + RefArrays.toString(inputDimensions));
    }
    final CudaTensorList tensorList = CudaSystem
        .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor tensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device,
              true);
          CudaTensorList temp_43_0002 = getOutput(gpu, tensor.addRef());
          tensor.freeRef();
          return temp_43_0002;
        }, inputData.addRef()), inputData.addRef());
    try {
      try {
        try {
          return new Result(tensorList, new Result.Accumulator() {
            {
              Result.addRefs(inObj);
              inputData.addRef();
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
              @Nonnull final int[] outputDimensions = {1, 1, inputDimensions[2] * inputDimensions[2]};
              if (!RefArrays.equals(delta.getDimensions(), outputDimensions)) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_43_0009 = new AssertionError(
                    RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
                delta.freeRef();
                throw temp_43_0009;
              }
              if (inObj[0].isAlive()) {
                final TensorList passbackTensorList = CudaSystem
                    .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                          @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(),
                              precision, MemoryType.Device, true);
                          CudaTensor deltaTensor = gpu.getTensor(delta.addRef(), precision,
                              MemoryType.Device, true);
                          CudaTensorList temp_43_0004 = GramianLayer.this.getFeedback(gpu,
                              inputTensor.addRef(),
                              deltaTensor.addRef());
                          deltaTensor.freeRef();
                          inputTensor.freeRef();
                          return temp_43_0004;
                        }, delta.addRef(), inputData.addRef()),
                        delta.addRef());
                inObj[0].accumulate(buffer == null ? null : buffer.addRef(),
                    passbackTensorList == null ? null : passbackTensorList.addRef());
                if (null != passbackTensorList)
                  passbackTensorList.freeRef();
              }
              delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public @SuppressWarnings("unused")
            void _free() {
              ReferenceCounting.freeRefs(inObj);
              inputData.freeRef();
            }
          }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                boolean temp_43_0005 = x.isAlive();
                x.freeRef();
                return temp_43_0005;
              });
            }

            @Override
            public final void accumulate(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
              Result.Accumulator temp_43_0010 = getAccumulator();
              assert temp_43_0010 != null;
              temp_43_0010.accept(buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef());
              temp_43_0010.freeRef();
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
        if (null != tensorList)
          tensorList.freeRef();
      }
    } finally {
      inputData.freeRef();
    }

  }

  @Nonnull
  public CudaTensorList getFeedback(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor inputTensor, @Nonnull final CudaTensor deltaTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());
    CudaMemory deltaMemory = deltaTensor.getMemory(gpu.addRef());
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
    assert inputMemory != null;
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

    deltaTensor.freeRef();
    RefIntStream.range(0, bands).forEach(RefUtil.wrapInterface(band -> {
          assert deltaMemory != null;
          CudaMemory deltaView1 = deltaMemory.withByteOffset(band * precision.size * bands);
          CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
              inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor1.getPtr(),
              deltaView1.getPtr(), precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
          RefUtil.freeRef(inputMemory.dirty());
          RefUtil.freeRef(deltaView1.dirty());
          deltaView1.freeRef();
          RefUtil.freeRef(bufferMemory.dirty());
          CudaMemory deltaView2 = deltaMemory.withByteOffset(band * precision.size);
          CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
              inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor2.getPtr(),
              deltaView2.getPtr(), precision.getPointer(1.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
          RefUtil.freeRef(inputMemory.dirty());
          RefUtil.freeRef(deltaView2.dirty());
          deltaView2.freeRef();
          RefUtil.freeRef(bufferMemory.dirty());
          CudaMemory outputViewMem = outputMemory.withByteOffset(bandDescriptor.cStride * band * precision.size);
          gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
              workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(), bufferMemory.getPtr(),
              precision.getPointer(0.0), bandDescriptor.getPtr(), outputViewMem.getPtr());
          RefUtil.freeRef(outputViewMem.dirty());
          outputViewMem.freeRef();
          RefUtil.freeRef(bufferMemory.dirty());
        }, workspacePtr, bufferDescriptor,
        inputTensor.addRef(), bufferMemory.addRef(),
        viewDescriptor1, bandDescriptor,
        viewDescriptor2, inputMemory.addRef(),
        deltaMemory == null ? null : deltaMemory.addRef(),
        reduceAddDescriptor.addRef(),
        multiplyDescriptor, indexPtr,
        outputMemory.addRef(), gpu));

    inputTensor.freeRef();
    reduceAddDescriptor.freeRef();
    bufferMemory.freeRef();
    if (null != deltaMemory)
      deltaMemory.freeRef();
    inputMemory.freeRef();
    CudaTensorList temp_43_0006 = new CudaTensorList(new CudaTensor(outputMemory.addRef(),
        outputDescriptor, precision), length, inputDimensions, precision);
    outputMemory.freeRef();
    return temp_43_0006;
  }

  @Nonnull
  public CudaTensorList getOutput(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor inputTensor) {
    int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
    @Nonnull final int[] inputDimensions = {inputTensor.descriptor.width, inputTensor.descriptor.height,
        inputTensor.descriptor.channels};
    final int length = inputTensor.descriptor.batchCount;
    final int bands = inputDimensions[2];
    @Nonnull final int[] outputDimensions = {1, 1, bands * bands};

    CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());

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

    assert inputMemory != null;
    @Nonnull final CudaMemory workspacePtr = gpu.allocate(Math.max(outputMemory.size, inputMemory.size), MemoryType.Device,
        true);
    @Nonnull final CudaMemory indexPtr = gpu.allocate((long) 12 * length, MemoryType.Device, true);
    RefIntStream.range(0, inputDimensions[2]).forEach(RefUtil.wrapInterface(band -> {
          CudaMemory inputView = inputMemory.withByteOffset(band * precision.size * inputTensor.descriptor.cStride);
          CudaSystem.handle(
              gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0), inputTensor.descriptor.getPtr(),
                  inputMemory.getPtr(), precision.getPointer(1.0), inputViewDescriptor.getPtr(), inputView.getPtr(),
                  precision.getPointer(0.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
          RefUtil.freeRef(bufferMemory.dirty());
          RefUtil.freeRef(inputView.dirty());
          inputView.freeRef();
          RefUtil.freeRef(inputMemory.dirty());
          CudaMemory outputView = outputMemory.withByteOffset(band * precision.size * bands);
          CudaSystem.handle(gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
              workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(),
              bufferMemory.getPtr(), precision.getPointer(0.0), outputViewDescriptor.getPtr(), outputView.getPtr()));
          RefUtil.freeRef(outputView.dirty());
          outputView.freeRef();
          RefUtil.freeRef(bufferMemory.dirty());
        }, outputMemory.addRef(), bufferDescriptor,
        indexPtr, inputViewDescriptor,
        inputMemory.addRef(),
        reduceAddDescriptor.addRef(),
        outputViewDescriptor,
        multiplyDescriptor, inputTensor.addRef(),
        workspacePtr, bufferMemory.addRef(), gpu));

    inputTensor.freeRef();
    reduceAddDescriptor.freeRef();
    RefUtil.freeRef(outputMemory.dirty());
    RefUtil.freeRef(bufferMemory.dirty());
    bufferMemory.freeRef();
    RefUtil.freeRef(inputMemory.dirty());

    inputMemory.freeRef();
    CudaTensorList temp_43_0007 = new CudaTensorList(new CudaTensor(outputMemory.addRef(),
        ouputDescriptor, precision), length, outputDimensions, precision);
    outputMemory.freeRef();
    return temp_43_0007;
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
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GramianLayer addRef() {
    return (GramianLayer) super.addRef();
  }
}
