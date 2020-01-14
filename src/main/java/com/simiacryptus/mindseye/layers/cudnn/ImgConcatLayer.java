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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.IntUnaryOperator;

@SuppressWarnings("serial")
public class ImgConcatLayer extends LayerBase implements MultiPrecision<ImgConcatLayer> {

  private int maxBands = -1;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public ImgConcatLayer() {
  }

  protected ImgConcatLayer(@Nonnull final JsonObject json) {
    super(json);
    maxBands = json.get("maxBands").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
    this.parallel = json.get("parallel").getAsBoolean();
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class);
  }

  public int getMaxBands() {
    return maxBands;
  }

  @Nonnull
  public ImgConcatLayer setMaxBands(final int maxBands) {
    this.maxBands = maxBands;
    return this.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgConcatLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this.addRef();
  }

  public boolean isParallel() {
    return parallel;
  }

  @Nonnull
  public ImgConcatLayer setParallel(boolean parallel) {
    this.parallel = parallel;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgConcatLayer(json);
  }

  @Nonnull
  public static Tensor eval(@Nonnull final RefList<Tensor> featureImage) {
    ImgConcatLayer layer = new ImgConcatLayer();
    Result temp_31_0011 = layer.eval(featureImage.toArray(new Tensor[]{}));
    assert temp_31_0011 != null;
    TensorList data = temp_31_0011.getData();
    temp_31_0011.freeRef();
    featureImage.freeRef();
    layer.freeRef();
    Tensor temp_31_0001 = data.get(0);
    data.freeRef();
    return temp_31_0001;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgConcatLayer[] addRefs(@Nullable ImgConcatLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayer::addRef)
        .toArray((x) -> new ImgConcatLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgConcatLayer[][] addRefs(@Nullable ImgConcatLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayer::addRefs)
        .toArray((x) -> new ImgConcatLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer temp_31_0012 = getCompatibilityLayer();
      Result temp_31_0009 = temp_31_0012.eval(Result.addRefs(inObj));
      temp_31_0012.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_31_0009;
    }
    TensorList temp_31_0013 = inObj[0].getData();
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    int[] dimensions = temp_31_0013.getDimensions();
    temp_31_0013.freeRef();
    assert 3 == dimensions.length;
    @Nonnull final int[] outputDimensions = RefArrays.copyOf(dimensions, dimensions.length);
    TensorList temp_31_0014 = inObj[0].getData();
    final int length = temp_31_0014.length();
    temp_31_0014.freeRef();
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_31_0015 = x.getData();
      @Nonnull
      int[] d = temp_31_0015.getDimensions();
      temp_31_0015.freeRef();
      TensorList temp_31_0016 = x.getData();
      boolean temp_31_0002 = 3 == d.length && d[0] == outputDimensions[0] && d[1] == outputDimensions[1]
          && temp_31_0016.length() == length;
      temp_31_0016.freeRef();
      x.freeRef();
      return temp_31_0002;
    });
    outputDimensions[2] = RefArrays.stream(Result.addRefs(inObj)).mapToInt(x -> {
      TensorList temp_31_0017 = x.getData();
      int temp_31_0003 = temp_31_0017.getDimensions()[2];
      temp_31_0017.freeRef();
      x.freeRef();
      return temp_31_0003;
    }).sum();
    if (0 < maxBands && outputDimensions[2] > maxBands) {
      outputDimensions[2] = maxBands;
    }
    try {
      return new Result(CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
        final long outputSize = ((long) length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0]
            * precision.size);
        @Nonnull final CudaMemory cudaOutput = gpu.allocate(outputSize, MemoryType.Managed.ifEnabled(), true);
        RefIntStream stream = RefIntStream.range(0, inObj.length);
        //if (!CoreSettings.INSTANCE.isConservative() && parallel) stream = stream.parallel();
        stream.forEach(RefUtil.wrapInterface((IntConsumer) i -> {
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          final TensorList input = inObj[i].getData();
          @Nonnull final int[] inputDimensions = input.getDimensions();
          assert inputDimensions[0] == outputDimensions[0];
          assert inputDimensions[1] == outputDimensions[1];
          int bandOffset = RefIntStream.range(0, i).map(RefUtil.wrapInterface((IntUnaryOperator) j -> {
            TensorList data = inObj[j].getData();
            int dimension = data.getDimensions()[2];
            data.freeRef();
            return dimension;
          }, Result.addRefs(inObj))).sum();
          if (maxBands > 0)
            bandOffset = Math.min(bandOffset, maxBands);
          int inputBands = inputDimensions[2];
          if (maxBands > 0)
            inputBands = Math.min(inputBands, maxBands - bandOffset);
          if (inputBands > 0) {
            @Nullable final CudaTensor cudaInput = gpu.getTensor(input.addRef(), precision,
                MemoryType.Device, false);
            assert maxBands <= 0 || inputBands <= maxBands;
            assert inputBands <= inputDimensions[2];
            @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
                inputBands, outputDimensions[1], outputDimensions[0], //
                outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
                outputDimensions[1] * outputDimensions[0], //
                outputDimensions[0], //
                1);

            @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, length,
                inputBands, inputDimensions[1], inputDimensions[0], //
                cudaInput.descriptor.nStride, //
                cudaInput.descriptor.cStride, //
                cudaInput.descriptor.hStride, //
                cudaInput.descriptor.wStride);

            int byteOffset = outputDescriptor.cStride * bandOffset * precision.size;
            CudaMemory cudaInputMemory = cudaInput.getMemory(gpu.addRef());
            cudaInput.freeRef();
            assert cudaInputMemory != null;
            gpu.cudnnTransformTensor(precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInputMemory.getPtr(),
                precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(byteOffset));
            inputDescriptor.freeRef();
            outputDescriptor.freeRef();
            assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
            RefUtil.freeRef(cudaInputMemory.dirty());
            cudaInputMemory.freeRef();
            RefUtil.freeRef(cudaOutput.dirty());
          }
          input.freeRef();
        }, Result.addRefs(inObj), cudaOutput.addRef()));
        CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(precision, length, outputDimensions[2],
            outputDimensions[1], outputDimensions[0]);
        CudaTensorList temp_31_0004 = new CudaTensorList(new CudaTensor(cudaOutput,
            outDesc.addRef(), precision), length, outputDimensions, precision);
        outDesc.freeRef();
        return temp_31_0004;
      }, Result.addRefs(inObj)), RefArrays.stream(Result.addRefs(inObj)).map(Result::getData).toArray()),
          new Result.Accumulator() {
            {
              Result.addRefs(inObj);
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
              assert delta.getDimensions()[0] == outputDimensions[0];
              assert delta.getDimensions()[1] == outputDimensions[1];
              assert delta.getDimensions()[2] == outputDimensions[2];
              if (!RefArrays.equals(delta.getDimensions(), outputDimensions)) {
                if (null != buffer)
                  buffer.freeRef();
                AssertionError temp_31_0010 = new AssertionError(
                    RefArrays.toString(delta.getDimensions()) + " != " + RefArrays.toString(outputDimensions));
                delta.freeRef();
                throw temp_31_0010;
              }
              //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
              @Nonnull
              RefIntStream stream = RefIntStream.range(0, inObj.length);
              if (!CoreSettings.INSTANCE().isSingleThreaded() && parallel)
                stream = stream.parallel();
              stream.forEach(RefUtil.wrapInterface(i -> {
                    final Result input = inObj[i].addRef();
                    TensorList temp_31_0018 = input.getData();
                    int[] inputDimentions = temp_31_0018.getDimensions();
                    temp_31_0018.freeRef();
                    assert 3 == inputDimentions.length;
                    TensorList temp_31_0019 = input.getData();
                    assert delta.length() == temp_31_0019.length();
                    temp_31_0019.freeRef();
                    assert inputDimentions[0] == outputDimensions[0];
                    assert inputDimentions[1] == outputDimensions[1];
                    int bandOffset = RefIntStream.range(0, i).map(RefUtil.wrapInterface(j -> {
                      TensorList data = inObj[j].getData();
                      int dimension = data.getDimensions()[2];
                      data.freeRef();
                      return dimension;
                    }, Result.addRefs(inObj))).sum();
                    int inputBands = maxBands <= 0 ? inputDimentions[2]
                        : Math.min(inputDimentions[2], maxBands - bandOffset);
                    if (inputBands > 0 && input.isAlive()) {
                      assert inputBands <= inputDimentions[2];
                      assert inputBands <= outputDimensions[2];
                      final TensorList passbackTensorList = CudaSystem
                          .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                            final CudaTensor result;
                            synchronized (gpu) {
                              result = gpu.getTensor(delta.addRef(), precision, MemoryType.Device,
                                  true);
                            }
                            @Nullable final CudaTensor cudaDelta = result.addRef();
                            result.freeRef();
                            CudaMemory cudaDeltaMemory = cudaDelta.getMemory(gpu);
                            if (inputDimentions[2] == inputBands) {
                              @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision,
                                  length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                                  cudaDelta.descriptor.nStride, //
                                  cudaDelta.descriptor.cStride, //
                                  cudaDelta.descriptor.hStride, //
                                  cudaDelta.descriptor.wStride);
                              int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                              assert cudaDeltaMemory != null;
                              CudaMemory ptr = cudaDeltaMemory.withByteOffset(byteOffset);
                              CudaTensor cudaTensor = new CudaTensor(ptr.addRef(),
                                  viewDescriptor, precision);
                              ptr.freeRef();
                              cudaDelta.freeRef();
                              cudaDeltaMemory.freeRef();
                              CudaTensorList temp_31_0005 = new CudaTensorList(
                                  cudaTensor.addRef(), length, inputDimentions, precision);
                              cudaTensor.freeRef();
                              return temp_31_0005;
                            } else {
                              @Nonnull final CudaDevice.CudaTensorDescriptor passbackTransferDescriptor = gpu.newTensorDescriptor(
                                  precision, length, inputBands, inputDimentions[1], inputDimentions[0], //
                                  inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                                  inputDimentions[1] * inputDimentions[0], //
                                  inputDimentions[0], //
                                  1);
                              @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                                  length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                                  inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                                  inputDimentions[1] * inputDimentions[0], //
                                  inputDimentions[0], //
                                  1);
                              @Nonnull final CudaDevice.CudaTensorDescriptor deltaViewDescriptor = gpu.newTensorDescriptor(precision,
                                  length, inputBands, inputDimentions[1], inputDimentions[0], //
                                  cudaDelta.descriptor.nStride, //
                                  cudaDelta.descriptor.cStride, //
                                  cudaDelta.descriptor.hStride, //
                                  cudaDelta.descriptor.wStride);
                              @Nonnull final CudaMemory cudaBackprop = gpu.allocate(
                                  (long) passbackDescriptor.nStride * length * precision.size,
                                  MemoryType.Managed.ifEnabled(), inputBands == inputDimentions[2]);
                              int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                              assert cudaDeltaMemory != null;
                              gpu.cudnnTransformTensor(precision.getPointer(1.0), deltaViewDescriptor.getPtr(),
                                  cudaDeltaMemory.getPtr().withByteOffset(byteOffset), precision.getPointer(0.0),
                                  passbackTransferDescriptor.getPtr(), cudaBackprop.getPtr());
                              deltaViewDescriptor.freeRef();
                              passbackTransferDescriptor.freeRef();
                              RefUtil.freeRef(cudaBackprop.dirty());
                              RefUtil.freeRef(cudaDeltaMemory.dirty());
                              cudaDelta.freeRef();
                              cudaDeltaMemory.freeRef();
                              CudaTensorList temp_31_0006 = new CudaTensorList(
                                  new CudaTensor(cudaBackprop,
                                      passbackDescriptor, precision),
                                  length, inputDimentions, precision);
                              return temp_31_0006;
                            }
                          }, delta.addRef()));
                      input.accumulate(buffer == null ? null : buffer.addRef(),
                          passbackTensorList == null ? null : passbackTensorList.addRef());
                      if (null != passbackTensorList)
                        passbackTensorList.freeRef();
                    }
                    //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                    input.freeRef();
                  }, delta.addRef(), Result.addRefs(inObj),
                  buffer == null ? null : buffer.addRef()));
              delta.freeRef();
              if (null != buffer)
                buffer.freeRef();
            }

            public @SuppressWarnings("unused")
            void _free() {
              ReferenceCounting.freeRefs(inObj);
            }
          }) {

        {
          Result.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
            boolean temp_31_0007 = x.isAlive();
            x.freeRef();
            return temp_31_0007;
          });
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
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
  ImgConcatLayer addRef() {
    return (ImgConcatLayer) super.addRef();
  }

}
