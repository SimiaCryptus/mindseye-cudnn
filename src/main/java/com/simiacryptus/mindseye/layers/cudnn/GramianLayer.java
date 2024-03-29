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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Gramian layer.
 */
@SuppressWarnings("serial")
public class GramianLayer extends LayerBase implements MultiPrecision {
  private static final Logger log = LoggerFactory.getLogger(GramianLayer.class);

  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private double alpha = 1.0;

  /**
   * Instantiates a new Gramian layer.
   */
  public GramianLayer() {
  }

  /**
   * Instantiates a new Gramian layer.
   *
   * @param id the id
   */
  public GramianLayer(UUID id) {
    super(id, "Gramian");
  }

  /**
   * Instantiates a new Gramian layer.
   *
   * @param json the json
   */
  protected GramianLayer(@Nonnull final JsonObject json) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    this.alpha = json.getAsJsonPrimitive("alpha").getAsDouble();
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   */
  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * From json gramian layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the gramian layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static GramianLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GramianLayer(json);
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
    final CudaTensorList tensorList = fwd(inputData.addRef());
    Result.Accumulator accumulator = new Accumulator(inputData, inputDimensions, GramianLayer.this.precision, GramianLayer.this.alpha, inObj[0].getAccumulator(), inObj[0].isAlive());
    boolean isAlive = Result.anyAlive(inObj);
    return new Result(tensorList, accumulator, isAlive);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GramianLayer addRef() {
    return (GramianLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList fwd(TensorList inputData) {
    return CudaSystem
        .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
          int pixels = inputTensor.descriptor.height * inputTensor.descriptor.width;
          @Nonnull final int[] inputDimensions = {inputTensor.descriptor.width, inputTensor.descriptor.height,
              inputTensor.descriptor.channels};
          final int length = inputTensor.descriptor.batchCount;
          final int bands = inputDimensions[2];
          @Nonnull final int[] outputDimensions = {1, 1, bands * bands};

          CudaMemory inputMemory = inputTensor.getMemory(gpu.addRef());

          final CudaDevice.CudaTensorDescriptor ouputDescriptor = gpu.newTensorDescriptor(precision, length, bands * bands, 1,
              1, bands * bands, //
              1, //
              1, //
              1);
          @Nullable final CudaMemory outputMemory = gpu.allocate((long) ouputDescriptor.nStride * precision.size * length,
              MemoryType.Device, true);

          final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
              inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
              inputDimensions[0] * inputDimensions[1], //
              inputDimensions[0], //
              1);
          @Nullable final CudaMemory bufferMemory = gpu.allocate((long) bufferDescriptor.nStride * length * precision.size,
              MemoryType.Device, true);

          final CudaDevice.CudaTensorDescriptor inputViewDescriptor = gpu.newTensorDescriptor(precision, length, 1,
              inputDimensions[1], inputDimensions[0], inputTensor.descriptor.nStride, //
              inputTensor.descriptor.cStride, //
              inputTensor.descriptor.hStride, //
              inputTensor.descriptor.wStride);

          CudaResource<cudnnReduceTensorDescriptor> reduceAddDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
              cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
              cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);

          final CudaDevice.CudaTensorDescriptor outputViewDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1,
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
                bufferMemory.dirty();
                inputView.dirty();
                inputView.freeRef();
                inputMemory.dirty();
                CudaMemory outputView = outputMemory.withByteOffset(band * precision.size * bands);
                CudaSystem.handle(gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size,
                    workspacePtr.getPtr(), workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(),
                    bufferMemory.getPtr(), precision.getPointer(0.0), outputViewDescriptor.getPtr(), outputView.getPtr()));
                outputView.dirty();
                outputView.freeRef();
                bufferMemory.dirty();
              }, outputMemory.addRef(), bufferDescriptor,
              indexPtr, inputViewDescriptor,
              inputMemory.addRef(),
              reduceAddDescriptor,
              outputViewDescriptor,
              multiplyDescriptor, inputTensor,
              workspacePtr, bufferMemory.addRef(), gpu));

          outputMemory.dirty();
          bufferMemory.dirty();
          bufferMemory.freeRef();
          inputMemory.dirty();
          inputMemory.freeRef();
          return new CudaTensorList(
              new CudaTensor(outputMemory, ouputDescriptor, precision),
              length, outputDimensions, precision);
        }, inputData.addRef()), inputData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final int[] inputDimensions;
    private Precision precision;
    private double alpha;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputData       the input data
     * @param inputDimensions the input dimensions
     * @param precision       the precision
     * @param alpha           the alpha
     * @param accumulator     the accumulator
     * @param alive           the alive
     */
    public Accumulator(TensorList inputData, int[] inputDimensions, Precision precision, double alpha, Result.Accumulator accumulator, boolean alive) {
      this.inputData = inputData;
      this.inputDimensions = inputDimensions;
      this.precision = precision;
      this.alpha = alpha;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      @Nonnull final int[] outputDimensions = {1, 1, inputDimensions[2] * inputDimensions[2]};
      int[] deltaDimensions = delta.getDimensions();
      if (!RefArrays.equals(deltaDimensions, outputDimensions)) {
        if (null != buffer)
          buffer.freeRef();
        delta.freeRef();
        throw new AssertionError(
            RefArrays.toString(deltaDimensions) + " != " + RefArrays.toString(outputDimensions));
      }
      if (alive) {
        this.accumulator.accept(buffer, CudaSystem
            .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
              CudaTensor inputCuda = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
              CudaTensor deltaCuda = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
              return getFeedback(
                  gpu,
                  inputCuda,
                  deltaCuda
              );
            }, delta.addRef(), inputData.addRef()), delta));
      } else {
        delta.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      inputData.freeRef();
    }

    /**
     * Gets feedback.
     *
     * @param gpu         the gpu
     * @param inputTensor the input tensor
     * @param deltaTensor the delta tensor
     * @return the feedback
     */
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
      final CudaDevice.CudaTensorDescriptor bufferDescriptor = gpu.newTensorDescriptor(precision, length, bands,
          inputDimensions[1], inputDimensions[0], inputDimensions[0] * inputDimensions[1] * bands, //
          inputDimensions[0] * inputDimensions[1], //
          inputDimensions[0], //
          1);
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands,
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

      final CudaDevice.CudaTensorDescriptor bandDescriptor = gpu.newTensorDescriptor(precision, length, 1,
          inputDimensions[1], inputDimensions[0], inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
          inputDimensions[1] * inputDimensions[0], inputDimensions[0], 1);
      final CudaDevice.CudaTensorDescriptor viewDescriptor1 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
          deltaTensor.descriptor.nStride, //
          deltaTensor.descriptor.cStride, //
          deltaTensor.descriptor.hStride, //
          deltaTensor.descriptor.wStride);
      final CudaDevice.CudaTensorDescriptor viewDescriptor2 = gpu.newTensorDescriptor(precision, length, bands, 1, 1, //
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
            inputMemory.dirty();
            deltaView1.dirty();
            deltaView1.freeRef();
            bufferMemory.dirty();
            CudaMemory deltaView2 = deltaMemory.withByteOffset(band * precision.size);
            CudaSystem.handle(gpu.cudnnOpTensor(multiplyDescriptor.getPtr(), precision.getPointer(1.0),
                inputTensor.descriptor.getPtr(), inputMemory.getPtr(), precision.getPointer(1.0), viewDescriptor2.getPtr(),
                deltaView2.getPtr(), precision.getPointer(1.0), bufferDescriptor.getPtr(), bufferMemory.getPtr()));
            inputMemory.dirty();
            deltaView2.dirty();
            deltaView2.freeRef();
            bufferMemory.dirty();
            CudaMemory outputViewMem = outputMemory.withByteOffset(bandDescriptor.cStride * band * precision.size);
            gpu.cudnnReduceTensor(reduceAddDescriptor.getPtr(), indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(),
                workspacePtr.size, precision.getPointer(alpha / pixels), bufferDescriptor.getPtr(), bufferMemory.getPtr(),
                precision.getPointer(0.0), bandDescriptor.getPtr(), outputViewMem.getPtr());
            outputViewMem.dirty();
            outputViewMem.freeRef();
            bufferMemory.dirty();
          }, workspacePtr, bufferDescriptor,
          inputTensor, bufferMemory,
          viewDescriptor1, bandDescriptor,
          viewDescriptor2, inputMemory,
          deltaMemory,
          reduceAddDescriptor,
          multiplyDescriptor, indexPtr,
          outputMemory.addRef(), gpu));

      return new CudaTensorList(
          new CudaTensor(outputMemory, outputDescriptor, precision),
          length, inputDimensions, precision);
    }
  }
}
