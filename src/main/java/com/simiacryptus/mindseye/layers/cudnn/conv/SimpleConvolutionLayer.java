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

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.ImgCropLayer;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.*;
import jcuda.jcudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public @RefAware
class SimpleConvolutionLayer extends LayerBase
    implements MultiPrecision<SimpleConvolutionLayer> {

  static final Logger log = LoggerFactory.getLogger(SimpleConvolutionLayer.class);
  public final Tensor kernel;
  @Nullable
  private final RefMap<Integer, CudaMemory> gpuFilters = new RefConcurrentHashMap<>();
  private int paddingX;
  private int paddingY;
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private int strideX = 1;
  private int strideY = 1;

  protected SimpleConvolutionLayer() {
    this(null);
  }

  public SimpleConvolutionLayer(final int width, final int height, final int bands) {
    this(new Tensor(width, height, bands));
  }

  protected SimpleConvolutionLayer(@Nonnull final JsonObject json,
                                   Map<CharSequence, byte[]> resources) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"), resources);
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    setPaddingX(json.get("paddingX").getAsInt());
    setPaddingY(json.get("paddingY").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  protected SimpleConvolutionLayer(@Nonnull final Tensor kernel) {
    super();
    @Nonnull
    int[] kernelSize = kernel.getDimensions();
    if (kernelSize.length != 3)
      throw new IllegalArgumentException();
    if (kernelSize[0] <= 0)
      throw new IllegalArgumentException();
    if (kernelSize[1] <= 0)
      throw new IllegalArgumentException();
    if (kernelSize[2] <= 0)
      throw new IllegalArgumentException();
    //    if (kernelSize[0] >= 60000) throw new IllegalArgumentException();
    //    if (kernelSize[1] >= 60000) throw new IllegalArgumentException();
    //    if (kernelSize[2] >= 60000) throw new IllegalArgumentException();
    this.kernel = kernel;
    this.setPaddingX((int) Math.ceil((kernelSize[0] - 1) / 2.0));
    this.setPaddingY((int) Math.ceil((kernelSize[1] - 1) / 2.0));

  }

  public SimpleConvolutionLayer(int width, int height, int inputBands, int outputBands) {
    this(width, height, inputBands * outputBands);
  }

  public Tensor getKernel() {
    return kernel;
  }

  public int[] getKernelDimensions() {
    return kernel.getDimensions();
  }

  public int getPaddingX() {
    return paddingX;
  }

  @Nonnull
  public SimpleConvolutionLayer setPaddingX(int paddingX) {
    this.paddingX = paddingX;
    return this;
  }

  public int getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public SimpleConvolutionLayer setPaddingY(int paddingY) {
    this.paddingY = paddingY;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public SimpleConvolutionLayer setPrecision(final Precision precision) {
    clearCudaFilters();
    this.precision = precision;
    return this;
  }

  public int getStrideX() {
    return strideX;
  }

  @Nonnull
  public SimpleConvolutionLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this;
  }

  public int getStrideY() {
    return strideY;
  }

  @Nonnull
  public SimpleConvolutionLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this;
  }

  @Nonnull
  public SimpleConvolutionLayer setWeightsLog(double f) {
    return set(() -> Math.pow(10, f) * (Math.random() - 0.5));
  }

  @SuppressWarnings("unused")
  public static SimpleConvolutionLayer fromJson(@Nonnull final JsonObject json,
                                                Map<CharSequence, byte[]> rs) {
    return new SimpleConvolutionLayer(json, rs);
  }

  @Nonnull
  public static int[] reverse(@Nonnull int... array) {
    for (int i = 0; i < array.length / 2; i++) {
      int j = array[array.length - (i + 1)];
      array[array.length - (i + 1)] = array[i];
      array[i] = j;
    }
    return array;
  }

  public static @SuppressWarnings("unused")
  SimpleConvolutionLayer[] addRefs(SimpleConvolutionLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleConvolutionLayer::addRef)
        .toArray((x) -> new SimpleConvolutionLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SimpleConvolutionLayer[][] addRefs(SimpleConvolutionLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleConvolutionLayer::addRefs)
        .toArray((x) -> new SimpleConvolutionLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assertAlive();
    @Nonnull final int[] rawInputDims = inObj[0].getData().getDimensions();
    final int[] kernelDimensions = kernel.getDimensions();
    int kernelLength = kernel.length();
    double[] kernelData = kernel.getData();
    int correctionX = correct(rawInputDims[0], strideX, kernelDimensions[0]);
    int correctionY = correct(rawInputDims[1], strideY, kernelDimensions[1]);
    int paddingX = Math.max(0, SimpleConvolutionLayer.this.paddingX - ((correctionX + 1) / 2));
    int paddingY = Math.max(0, SimpleConvolutionLayer.this.paddingY - ((correctionY + 1) / 2));
    if (correctionX >= kernelDimensions[0])
      correctionX -= kernelDimensions[0];
    if (correctionY >= kernelDimensions[1])
      correctionY -= kernelDimensions[1];
    assert paddingX >= 0;
    assert paddingY >= 0;
    assert correctionX >= 0;
    assert correctionY >= 0;
    @Nullable
    Result input;
    if (correctionX > 0 || correctionY > 0) {
      @Nonnull
      ImgCropLayer imgCropLayer = new ImgCropLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY)
          .setPrecision(precision).setHorizontalAlign(ImgCropLayer.Alignment.Center)
          .setVerticalAlign(ImgCropLayer.Alignment.Center).setRoundUp(false);
      input = imgCropLayer.eval(inObj[0]);
    } else {
      input = inObj[0];
    }
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    final int inputLength = inputData.length();
    final int[] outputSize = getOutputSize(inputDims);
    CudaTensorList run = CudaSystem.run(gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert 0 < kernelLength;
      assert kernelDimensions[0] * kernelDimensions[1] * kernelDimensions[2] == kernelLength;
      return fwd(gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, inputData,
          getCudaFilter(gpu));
    }, inputData);

    return new Result(run, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      delta.assertAlive();
      buffer.assertAlive();
      inputData.assertAlive();
      assert delta.length() == inputLength;
      Runnable learnFn = () -> {
        if (!isFrozen()) {
          @Nonnull final Tensor weightGradient = CudaSystem.run(gpu -> {
            @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
            @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
            final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
                cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1],
                kernelDimensions[0]);
            final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
                cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
            final int backwardFilterAlgorithm = getBackwardFilterAlgorithm(gpu, deltaTensor, inputTensor,
                filterDescriptor, convolutionDescriptor);
            final CudaMemory backwardsFilterWorkSpace = gpu.allocateBackwardFilterWorkspace(
                inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(),
                deltaTensor.descriptor.getPtr(), backwardFilterAlgorithm, 1);
            @Nonnull
            CudaMemory filterPtr = gpu.allocate((long) kernelLength * precision.size, MemoryType.Device, true);
            try {
              CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
              CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu, MemoryType.Managed.ifEnabled());
              //              inputTensorMemory.synchronize();
              CudaSystem.handle(gpu.cudnnConvolutionBackwardFilter(precision.getPointer(1.0),
                  inputTensor.descriptor.getPtr(), inputTensorMemory.getPtr(), deltaTensor.descriptor.getPtr(),
                  deltaTensorMemory.getPtr(), convolutionDescriptor.getPtr(), backwardFilterAlgorithm,
                  backwardsFilterWorkSpace.getPtr(), backwardsFilterWorkSpace.size, precision.getPointer(0.0),
                  filterDescriptor.getPtr(), filterPtr.getPtr()));
              filterPtr.dirty();
              deltaTensorMemory.dirty();
              inputTensorMemory.dirty();
              backwardsFilterWorkSpace.dirty();
              return filterPtr.read(precision, kernelDimensions);
            } catch (@Nonnull final Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s",
                  RefArrays.toString(inputDims),
                  RefArrays.toString(kernelDimensions),
                  RefArrays.toString(outputSize)), e);
            }
          }, delta);
          buffer.get(SimpleConvolutionLayer.this.getId(), kernelData).addInPlace(weightGradient.getData());
          clearCudaFilters();
        }
      };
      Runnable backpropFn = () -> {
        if (input.isAlive()) {
          final TensorList inputBufferTensors = CudaSystem.run(gpu -> {
            return bck_workaround(gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, delta);
          }, delta);
          if (null != inputBufferTensors) {
            input.accumulate(buffer, inputBufferTensors);
          }
        }
      };
      RefStream.of(learnFn, backpropFn).forEach(Runnable::run);
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      @Override
      public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

      public void _free() {
      }
    };
  }

  public CudaTensorList bck_workaround(CudnnHandle gpu, int paddingX, int paddingY, int inputLength, int[] inputDims,
                                       int[] kernelDimensions, int[] outputSize, TensorList delta) {
    if (1 != kernelDimensions[0] || 1 != kernelDimensions[1]) {
      return bck(gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, delta,
          getCudaFilter(gpu));
    } else {
      return fwd(gpu, paddingX, paddingY, inputLength, outputSize, new int[]{1, 1, outputSize[2], inputDims[2]},
          inputDims, delta,
          gpu.allocate((long) kernel.length() * precision.size, MemoryType.Device, true).write(precision,
              kernel.reshapeCast(1, 1, inputDims[2], outputSize[2]).permuteDimensions(0, 1, 3, 2).getData()));
    }
  }

  public CudaTensorList bck(CudnnHandle gpu, int paddingX, int paddingY, int inputLength, int[] inputDims,
                            int[] kernelDimensions, int[] outputSize, TensorList delta, CudaMemory filterPtr) {
    final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, inputLength,
        inputDims[2], inputDims[1], inputDims[0], inputDims[2] * inputDims[1] * inputDims[0],
        inputDims[1] * inputDims[0], inputDims[0], 1);
    final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
        cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1], kernelDimensions[0]);
    final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
        cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
    @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
    final int backwardDataAlgorithm = getBackwardDataAlgorithm(gpu, deltaTensor.descriptor, filterDescriptor,
        convolutionDescriptor, inputDescriptor);
    final CudaMemory backwardsDataWorkSpace = gpu.allocateBackwardDataWorkspace(inputDescriptor.getPtr(),
        filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(),
        backwardDataAlgorithm, 1);
    try {
      @Nonnull final CudaMemory passbackMemory = gpu.allocate((long) Tensor.length(inputDims) * inputLength * precision.size,
          MemoryType.Managed.ifEnabled(), true);
      CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
      //              deltaTensorMemory.synchronize();
      CudaSystem.handle(gpu.cudnnConvolutionBackwardData(precision.getPointer(1.0), filterDescriptor.getPtr(),
          filterPtr.getPtr(), deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
          convolutionDescriptor.getPtr(), backwardDataAlgorithm, backwardsDataWorkSpace.getPtr(),
          backwardsDataWorkSpace.size, precision.getPointer(0.0), inputDescriptor.getPtr(), passbackMemory.getPtr()));
      passbackMemory.dirty();
      backwardsDataWorkSpace.dirty();
      deltaTensorMemory.dirty();
      //              deltaTensorMemory.synchronize();
      filterPtr.dirty();
      return new CudaTensorList(new CudaTensor(passbackMemory, inputDescriptor, precision), inputLength, inputDims,
          precision);
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException(String.format("Error in convolution %s x %s => %s",
          RefArrays.toString(inputDims),
          RefArrays.toString(kernelDimensions),
          RefArrays.toString(outputSize)), e);
    }
  }

  public CudaTensorList fwd(CudnnHandle gpu, int paddingX, int paddingY, int inputLength, int[] inputDims,
                            int[] kernelDimensions, int[] outputSize, TensorList inputData, CudaMemory filterPtr) {
    CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
        cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
    CudaTensor tensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
    CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
        cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1], kernelDimensions[0]);
    String msg = String.format("Error in convolution %s x %s",
        RefArrays.toString(inputDims),
        RefArrays.toString(kernelDimensions));
    final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, inputLength,
        inputDims[2], inputDims[1], inputDims[0], inputDims[2] * inputDims[1] * inputDims[0],
        inputDims[1] * inputDims[0], inputDims[0], 1);
    final int[] outputDims = RefIntStream.of(reverse(
        CudaSystem.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr())))
        .limit(3).toArray();
    final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, inputLength,
        outputDims[2], outputDims[1], outputDims[0], outputDims[2] * outputDims[1] * outputDims[0],
        outputDims[1] * outputDims[0], outputDims[0], 1);
    final int forwardAlgorithm = getForwardAlgorithm(gpu, tensor, filterDescriptor, convolutionDescriptor,
        outputDescriptor);
    final CudaMemory forwardWorkspace = gpu.allocateForwardWorkspace(inputDescriptor.getPtr(),
        filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), forwardAlgorithm, 1);
    try {
      @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) Tensor.length(outputDims) * inputLength * precision.size,
          MemoryType.Managed.ifEnabled(), true);
      CudaMemory inputTensorMemory = tensor.getMemory(gpu);
      //        inputTensorMemory.synchronize();
      CudaSystem.handle(gpu.cudnnConvolutionForward(precision.getPointer(1.0), inputDescriptor.getPtr(),
          inputTensorMemory.getPtr(), filterDescriptor.getPtr(), filterPtr.getPtr(), convolutionDescriptor.getPtr(),
          forwardAlgorithm, null == forwardWorkspace ? null : forwardWorkspace.getPtr(),
          null == forwardWorkspace ? 0 : forwardWorkspace.size, precision.getPointer(0.0), outputDescriptor.getPtr(),
          outputBuffer.getPtr()));
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      forwardWorkspace.dirty();
      filterPtr.dirty();
      outputBuffer.dirty();
      inputTensorMemory.dirty();
      return new CudaTensorList(new CudaTensor(outputBuffer, outputDescriptor, precision), inputLength, outputDims,
          precision);
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException(msg, e);
    }
  }

  public int getForwardAlgorithm(final CudnnHandle gpu, final CudaTensor inputTensor,
                                 final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                 final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                 final CudaDevice.CudaTensorDescriptor outputDescriptor) {
    //    return cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    return gpu.getForwardAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), outputDescriptor.getPtr(),
        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
  }

  public int getBackwardFilterAlgorithm(final CudnnHandle gpu, final CudaTensor deltaTensor,
                                        final CudaTensor inputTensor, final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                        final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    return gpu.getBackwardFilterAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(),
        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
  }

  public int getBackwardDataAlgorithm(final CudnnHandle gpu, final CudaDevice.CudaTensorDescriptor dyDescriptor,
                                      final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                      final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                      CudaDevice.CudaTensorDescriptor dxDescriptor) {
    return cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    //    return gpu.getBackwardDataAlgorithm(
    //        dyDescriptor.getPtr(),
    //        filterDescriptor.getPtr(),
    //        convolutionDescriptor.getPtr(),
    //        dxDescriptor.getPtr(),
    //        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
  }

  public long evictDeviceData(final int deviceId) {
    CudaMemory remove = gpuFilters.remove(deviceId);
    if (null != remove) {
      if (1 == remove.currentRefCount()) {
        return remove.size;
      } else {
        CudaMemory race = gpuFilters.put(deviceId, remove);
        return 0;
      }
    } else {
      return 0;
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    JsonElement value;
    try {
      value = kernel.getJson(resources, dataSerializer);
    } catch (Throwable e) {
      throw new RuntimeException("Error serializing convolution"
          + RefArrays.toString(this.kernel.getDimensions()), e);
    }
    json.add("filter", value);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("precision", precision.name());
    return json;
  }

  public int[] getOutputSize(final int... inputSize) {
    @Nonnull final int[] kernelSize = kernel.getDimensions();
    try {
      return RefIntStream.range(0, kernelSize.length).map(i -> {
        int x;
        if (i == kernelSize.length - 1) {
          //assert kernelSize[i] == inputSize[i];
          x = kernelSize[i] / inputSize[i];
        } else {
          int padding;
          if (i == 0) {
            padding = this.paddingX;
          } else if (i == 1) {
            padding = this.paddingY;
          } else {
            throw new IllegalStateException();
          }
          x = inputSize[i] - (kernelSize[i] - 1) + padding * 2;
        }
        assert 0 < x;
        return x;
      }).toArray();
    } catch (Throwable e) {
      throw new RuntimeException(String.format("Error apply convolution %s x %s (%s)",
          RefArrays.toString(inputSize),
          RefArrays.toString(kernelSize), getName()), e);
    }
  }

  @Nonnull
  public SimpleConvolutionLayer set(@Nonnull final DoubleSupplier f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
    return this;
  }

  @Nonnull
  public SimpleConvolutionLayer set(@Nonnull final ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  @Nonnull
  public void setPaddingXY(int x, int y) {
    setPaddingX(x).setPaddingY(y);
  }

  public void set(@Nonnull Tensor kernel) {
    this.kernel.set(kernel);
  }

  @Override
  public boolean assertAlive() {
    if (!super.assertAlive()) {
      assert false;
      return false;
    }
    if (!kernel.assertAlive()) {
      assert false;
      return false;
    }
    return true;
  }

  public SimpleConvolutionLayer explode() {
    return this;
  }

  public SimpleConvolutionLayer setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
    return this;
  }

  public void _free() {
    clearCudaFilters();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleConvolutionLayer addRef() {
    return (SimpleConvolutionLayer) super.addRef();
  }

  private int correct(int dim, int modulus, int offset) {
    int adj = modulus - ((dim - offset) % modulus);
    while (adj < 0)
      adj += modulus;
    while (adj >= modulus)
      adj -= modulus;
    return adj;
  }

  @Nonnull
  private CudaMemory getCudaFilter(final CudaDevice deviceNumber) {
    return CudaSettings.INSTANCE().isConvolutionCache() ? getCudaFilter_cached(deviceNumber)
        : getCudaFilter_instance(deviceNumber);
  }

  @Nonnull
  private CudaMemory getCudaFilter_instance(final CudaDevice device) {
    double[] data = kernel.getData();
    return device.allocate((long) data.length * precision.size, MemoryType.Device, true).write(precision, data);
  }

  @Nonnull
  private CudaMemory getCudaFilter_cached(final CudaDevice device) {
    CudaMemory cudaMemory = gpuFilters.get(device.getDeviceId());
    if (null != cudaMemory && cudaMemory.tryAddRef()) {
      return cudaMemory;
    }
    CudaMemory newInstance = getCudaFilter_instance(device);
    CudaMemory replaced = gpuFilters.put(device.getDeviceId(), newInstance);
    return newInstance;
  }

  private void clearCudaFilters() {
    gpuFilters.keySet().stream().collect(RefCollectors.toList()).stream()
        .forEach(gpuFilters::remove);
  }
}
