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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class SimpleConvolutionLayer extends LayerBase implements MultiPrecision {

  static final Logger log = LoggerFactory.getLogger(SimpleConvolutionLayer.class);
  @Nonnull
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

  protected SimpleConvolutionLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"), resources);
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    setPaddingX(json.get("paddingX").getAsInt());
    setPaddingY(json.get("paddingY").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
    ObjectRegistry.register(addRef());
  }

  protected SimpleConvolutionLayer(@Nonnull final Tensor kernel) {
    super();
    @Nonnull
    int[] kernelSize = kernel.getDimensions();
    if (kernelSize.length != 3) {
      kernel.freeRef();
      throw new IllegalArgumentException();
    }
    if (kernelSize[0] <= 0) {
      kernel.freeRef();
      throw new IllegalArgumentException();
    }
    if (kernelSize[1] <= 0) {
      kernel.freeRef();
      throw new IllegalArgumentException();
    }
    if (kernelSize[2] <= 0) {
      kernel.freeRef();
      throw new IllegalArgumentException();
    }
    //    if (kernelSize[0] >= 60000) throw new IllegalArgumentException();
    //    if (kernelSize[1] >= 60000) throw new IllegalArgumentException();
    //    if (kernelSize[2] >= 60000) throw new IllegalArgumentException();
    this.kernel = kernel;
    setPaddingX((int) Math.ceil((kernelSize[0] - 1) / 2.0));
    setPaddingY((int) Math.ceil((kernelSize[1] - 1) / 2.0));
    ObjectRegistry.register(addRef());
  }

  public SimpleConvolutionLayer(int width, int height, int inputBands, int outputBands) {
    this(width, height, inputBands * outputBands);
  }

  @Nullable
  public Tensor getKernel() {
    return kernel.addRef();
  }

  @Nonnull
  public int[] getKernelDimensions() {
    return kernel.getDimensions();
  }

  public int getPaddingX() {
    return paddingX;
  }

  public void setPaddingX(int paddingX) {
    this.paddingX = paddingX;
  }

  public int getPaddingY() {
    return paddingY;
  }

  public void setPaddingY(int paddingY) {
    this.paddingY = paddingY;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public void setPrecision(final Precision precision) {
    clearCudaFilters();
    this.precision = precision;
  }

  public int getStrideX() {
    return strideX;
  }

  public void setStrideX(int strideX) {
    this.strideX = strideX;
  }

  public int getStrideY() {
    return strideY;
  }

  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }

  @Nonnull
  public SimpleConvolutionLayer setWeightsLog(double f) {
    set(() -> Math.pow(10, f) * (Math.random() - 0.5));
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SimpleConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assertAlive();
    TensorList data0 = inObj[0].getData();
    @Nonnull final int[] rawInputDims = data0.getDimensions();
    data0.freeRef();
    final int[] kernelDimensions = kernel.getDimensions();
    int kernelLength = kernel.length();
    double[] kernelData = kernel.getData();
    int correctionX = correct(rawInputDims[0], strideX, kernelDimensions[0]);
    int correctionY = correct(rawInputDims[1], strideY, kernelDimensions[1]);
    int paddingX = Math.max(0, this.paddingX - (correctionX + 1) / 2);
    int paddingY = Math.max(0, this.paddingY - (correctionY + 1) / 2);
    if (correctionX >= kernelDimensions[0])
      correctionX -= kernelDimensions[0];
    if (correctionY >= kernelDimensions[1])
      correctionY -= kernelDimensions[1];
    assert paddingX >= 0;
    assert paddingY >= 0;
    assert correctionX >= 0;
    assert correctionY >= 0;
    @Nullable
    Result input = null;
    if (correctionX > 0 || correctionY > 0) {
      ImgCropLayer imgCropLayer = new ImgCropLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY);
      imgCropLayer.setPrecision(precision);
      imgCropLayer.setHorizontalAlign(ImgCropLayer.Alignment.Center);
      imgCropLayer.setVerticalAlign(ImgCropLayer.Alignment.Center);
      imgCropLayer.setRoundUp(false);
      RefUtil.freeRef(input);
      input = imgCropLayer.eval(inObj[0].addRef());
      imgCropLayer.freeRef();
    } else {
      RefUtil.freeRef(input);
      input = inObj[0].addRef();
    }
    RefUtil.freeRef(inObj);
    assert input != null;
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    final int inputLength = inputData.length();
    final int[] outputSize = getOutputSize(inputDims);
    CudaTensorList data = fwd(kernelDimensions, kernelLength, paddingX, paddingY, inputData.addRef(), inputDims, inputLength, outputSize);

    final boolean inputAlive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(
        inputData,
        inputLength, outputSize,
        inputDims, kernelDimensions,
        paddingY, paddingX,
        kernelLength, kernelData,
        inputAlive,
        addRef(), input.getAccumulator()
    );
    input.freeRef();
    return new Result(data, accumulator, inputAlive || !isFrozen());
  }

  @Nullable
  public CudaTensorList bck_workaround(@Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, @Nonnull int[] inputDims,
                                       @Nonnull int[] kernelDimensions, @Nonnull int[] outputSize, @Nullable TensorList delta) {
    if (1 != kernelDimensions[0] || 1 != kernelDimensions[1]) {
      CudaMemory cudaFilter = getCudaFilter(gpu.addRef());
      return bck(gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, delta, cudaFilter);
    } else {
      CudaMemory cudaMemory = gpu.allocate((long) kernel.length() * precision.size, MemoryType.Device, true);
      Tensor temp_16_0023 = kernel.reshapeCast(1, 1, inputDims[2], outputSize[2]);
      Tensor temp_16_0024 = temp_16_0023.permuteDimensions(0, 1, 3, 2);
      temp_16_0023.freeRef();
      cudaMemory.write(precision, temp_16_0024.getData());
      CudaTensorList cudaTensorList = fwd(gpu, paddingX, paddingY, inputLength, outputSize,
          new int[]{1, 1, outputSize[2], inputDims[2]}, inputDims, delta,
          cudaMemory);
      temp_16_0024.freeRef();
      return cudaTensorList;
    }
  }

  @Nullable
  public CudaTensorList bck(@Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, @Nonnull int[] inputDims,
                            int[] kernelDimensions, int[] outputSize, @Nullable TensorList delta, @Nonnull CudaMemory filterPtr) {
    final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, inputLength,
        inputDims[2], inputDims[1], inputDims[0], inputDims[2] * inputDims[1] * inputDims[0],
        inputDims[1] * inputDims[0], inputDims[0], 1);
    final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
        cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1], kernelDimensions[0]);
    final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
        cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
    @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
    final int backwardDataAlgorithm = getBackwardDataAlgorithm(gpu.addRef(),
        deltaTensor.descriptor.addRef(),
        filterDescriptor.addRef(),
        convolutionDescriptor.addRef(),
        inputDescriptor.addRef());
    final CudaMemory backwardsDataWorkSpace = gpu.allocateBackwardDataWorkspace(inputDescriptor.getPtr(),
        filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(),
        backwardDataAlgorithm, 1);
    try {
      @Nonnull final CudaMemory passbackMemory = gpu.allocate((long) Tensor.length(inputDims) * inputLength * precision.size,
          MemoryType.Managed.ifEnabled(), true);
      CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef());
      //              deltaTensorMemory.synchronize();
      assert deltaTensorMemory != null;
      CudaSystem.handle(gpu.cudnnConvolutionBackwardData(precision.getPointer(1.0), filterDescriptor.getPtr(),
          filterPtr.getPtr(), deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
          convolutionDescriptor.getPtr(), backwardDataAlgorithm, backwardsDataWorkSpace.getPtr(),
          backwardsDataWorkSpace.size, precision.getPointer(0.0), inputDescriptor.getPtr(), passbackMemory.getPtr()));
      passbackMemory.dirty();
      backwardsDataWorkSpace.dirty();
      deltaTensorMemory.dirty();
      deltaTensorMemory.freeRef();
      filterPtr.dirty();
      CudaTensorList temp_16_0008 = new CudaTensorList(new CudaTensor(passbackMemory,
          inputDescriptor.addRef(), precision), inputLength, inputDims, precision);
      inputDescriptor.freeRef();
      filterDescriptor.freeRef();
      convolutionDescriptor.freeRef();
      deltaTensor.freeRef();
      backwardsDataWorkSpace.freeRef();
      filterPtr.freeRef();
      return temp_16_0008;
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException(RefString.format("Error in convolution %s x %s => %s", RefArrays.toString(inputDims),
          RefArrays.toString(kernelDimensions), RefArrays.toString(outputSize)), e);
    } finally {
      gpu.freeRef();
    }
  }

  @Nullable
  public CudaTensorList fwd(@Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, int[] inputDims,
                            int[] kernelDimensions, int[] outputSize, @Nullable TensorList inputData, @Nonnull CudaMemory filterPtr) {
    CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
        cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
    CudaTensor tensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
    CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
        cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1], kernelDimensions[0]);
    String msg = RefString.format("Error in convolution %s x %s", RefArrays.toString(inputDims),
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
    final int forwardAlgorithm = getForwardAlgorithm(gpu.addRef(), tensor.addRef(),
        filterDescriptor.addRef(),
        convolutionDescriptor.addRef(),
        outputDescriptor.addRef());
    final CudaMemory forwardWorkspace = gpu.allocateForwardWorkspace(inputDescriptor.getPtr(),
        filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), forwardAlgorithm, 1);
    try {
      @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) Tensor.length(outputDims) * inputLength * precision.size,
          MemoryType.Managed.ifEnabled(), true);
      CudaMemory inputTensorMemory = tensor.getMemory(gpu.addRef());
      //        inputTensorMemory.synchronize();
      assert inputTensorMemory != null;
      CudaSystem.handle(gpu.cudnnConvolutionForward(precision.getPointer(1.0), inputDescriptor.getPtr(),
          inputTensorMemory.getPtr(), filterDescriptor.getPtr(), filterPtr.getPtr(), convolutionDescriptor.getPtr(),
          forwardAlgorithm, forwardWorkspace.getPtr(),
          forwardWorkspace.size, precision.getPointer(0.0), outputDescriptor.getPtr(),
          outputBuffer.getPtr()));
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      forwardWorkspace.dirty();
      forwardWorkspace.freeRef();
      filterPtr.dirty();
      outputBuffer.dirty();
      inputTensorMemory.dirty();
      inputTensorMemory.freeRef();
      convolutionDescriptor.freeRef();
      tensor.freeRef();
      filterDescriptor.freeRef();
      inputDescriptor.freeRef();
      return new CudaTensorList(
          new CudaTensor(outputBuffer, outputDescriptor, precision),
          inputLength, outputDims, precision);
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException(msg, e);
    } finally {
      filterPtr.freeRef();
      gpu.freeRef();
    }
  }

  public int getForwardAlgorithm(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor inputTensor,
                                 @Nonnull final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                 @Nonnull final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                 @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor) {
    int gpuForwardAlgorithm = gpu.getForwardAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), outputDescriptor.getPtr(),
        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
    gpu.freeRef();
    outputDescriptor.freeRef();
    convolutionDescriptor.freeRef();
    filterDescriptor.freeRef();
    inputTensor.freeRef();
    //    return cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    return gpuForwardAlgorithm;
  }

  public int getBackwardFilterAlgorithm(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor deltaTensor,
                                        @Nonnull final CudaTensor inputTensor, @Nonnull final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                        @Nonnull final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    int backwardFilterAlgorithm = gpu.getBackwardFilterAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(),
        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
    convolutionDescriptor.freeRef();
    filterDescriptor.freeRef();
    inputTensor.freeRef();
    deltaTensor.freeRef();
    gpu.freeRef();
    return backwardFilterAlgorithm;
  }

  public int getBackwardDataAlgorithm(@Nonnull final CudnnHandle gpu, @Nullable final CudaDevice.CudaTensorDescriptor dyDescriptor,
                                      @Nullable final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                      @Nullable final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                      @Nullable CudaDevice.CudaTensorDescriptor dxDescriptor) {
    gpu.freeRef();
    if (null != dxDescriptor)
      dxDescriptor.freeRef();
    if (null != convolutionDescriptor)
      convolutionDescriptor.freeRef();
    if (null != filterDescriptor)
      filterDescriptor.freeRef();
    if (null != dyDescriptor)
      dyDescriptor.freeRef();
    return cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    //    return gpu.getBackwardDataAlgorithm(
    //        dyDescriptor.getPtr(),
    //        filterDescriptor.getPtr(),
    //        convolutionDescriptor.getPtr(),
    //        dxDescriptor.getPtr(),
    //        CudaSettings.INSTANCE().getConvolutionWorkspaceSizeLimit());
  }

  public long evictDeviceData(final int deviceId) {
    synchronized (gpuFilters) {
      assert gpuFilters != null;
      CudaMemory remove = gpuFilters.remove(deviceId);
      if (null != remove) {
        if (1 == remove.currentRefCount()) {
          long size = remove.size;
          remove.freeRef();
          return size;
        } else {
          RefUtil.freeRef(gpuFilters.put(deviceId, remove));
          return 0;
        }
      } else {
        return 0;
      }
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    JsonElement value;
    try {
      value = kernel.getJson(resources, dataSerializer);
    } catch (Throwable e) {
      throw new RuntimeException("Error serializing convolution" + RefArrays.toString(this.kernel.getDimensions()), e);
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
      throw new RuntimeException(RefString.format("Error apply convolution %s x %s (%s)", RefArrays.toString(inputSize),
          RefArrays.toString(kernelSize), getName()), e);
    }
  }

  public void set(@Nonnull DoubleSupplier f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
  }

  public void set(@Nonnull ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  @Nonnull
  public void setPaddingXY(int x, int y) {
    setPaddingX(x);
    setPaddingY(y);
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

  @Nonnull
  public void setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
  }

  public void _free() {
    super._free();
    clearCudaFilters();
    if (null != gpuFilters) {
      gpuFilters.freeRef();
    }
    kernel.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SimpleConvolutionLayer addRef() {
    return (SimpleConvolutionLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList fwd(int[] kernelDimensions, int kernelLength, int paddingX, int paddingY, TensorList inputData, int[] inputDims, int inputLength, int[] outputSize) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert 0 < kernelLength;
      assert kernelDimensions[0] * kernelDimensions[1] * kernelDimensions[2] == kernelLength;
      return fwd(gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize,
          inputData.addRef(), getCudaFilter(gpu.addRef()));
    }, inputData.addRef()), inputData);
  }

  private int correct(int dim, int modulus, int offset) {
    int adj = modulus - (dim - offset) % modulus;
    while (adj < 0)
      adj += modulus;
    while (adj >= modulus)
      adj -= modulus;
    return adj;
  }

  @Nonnull
  private CudaMemory getCudaFilter(@Nonnull final CudaDevice deviceNumber) {
    if (CudaSettings.INSTANCE().isConvolutionCache()) return getCudaFilter_cached(deviceNumber);
    else return getCudaFilter_instance(deviceNumber);
  }

  @Nonnull
  private CudaMemory getCudaFilter_instance(@Nonnull final CudaDevice device) {
    CudaMemory cudaMemory = device.allocate((long) kernel.length() * precision.size, MemoryType.Device, true);
    cudaMemory.write(precision, kernel.getData());
    device.freeRef();
    return cudaMemory;
  }

  @Nonnull
  @RefIgnore
  private CudaMemory getCudaFilter_cached(@Nonnull final CudaDevice device) {
    synchronized (gpuFilters) {
      assert gpuFilters != null;
      return gpuFilters.computeIfAbsent(device.getDeviceId(), deviceId -> getCudaFilter_instance(device));
    }
  }

  private void clearCudaFilters() {
    synchronized (gpuFilters) {
      assert gpuFilters != null;
      gpuFilters.clear();
    }
  }

  private class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final int inputLength;
    private final int[] outputSize;
    private final int[] inputDims;
    private final int[] kernelDimensions;
    private final int paddingY;
    private final int paddingX;
    private final int kernelLength;
    private final double[] kernelData;
    private final boolean alive;
    private UUID id;
    private SimpleConvolutionLayer simpleConvolutionLayer;
    private Result.Accumulator inputAccumulator;

    public Accumulator(TensorList inputData, int inputLength, int[] outputSize, int[] inputDims, int[] kernelDimensions, int paddingY, int paddingX, int kernelLength, double[] kernelData, boolean alive, SimpleConvolutionLayer simpleConvolutionLayer, Result.Accumulator accumulator) {
      this.inputData = inputData;
      this.inputLength = inputLength;
      this.outputSize = outputSize;
      this.inputDims = inputDims;
      this.kernelDimensions = kernelDimensions;
      this.paddingY = paddingY;
      this.paddingX = paddingX;
      this.kernelLength = kernelLength;
      this.kernelData = kernelData;
      this.alive = alive;
      this.id = simpleConvolutionLayer.getId();
      this.simpleConvolutionLayer = simpleConvolutionLayer;
      inputAccumulator = accumulator;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      delta.assertAlive();
      buffer.assertAlive();
      inputData.assertAlive();
      assert delta.length() == inputLength;
      Runnable learnFn = RefUtil.wrapInterface(() -> {
        if (!simpleConvolutionLayer.isFrozen()) {
          @Nonnull final Tensor weightGradient = CudaSystem
              .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, Tensor>) gpu -> {
                @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta.addRef(), precision,
                    MemoryType.Device, true);
                @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData.addRef(),
                    precision, MemoryType.Device, false);
                final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(
                    precision, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2],
                    kernelDimensions[1], kernelDimensions[0]);
                final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu
                    .newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY,
                        paddingX, strideY, strideX, 1, 1);
                final int backwardFilterAlgorithm = simpleConvolutionLayer.getBackwardFilterAlgorithm(
                    gpu.addRef(), deltaTensor.addRef(),
                    inputTensor.addRef(),
                    filterDescriptor.addRef(),
                    convolutionDescriptor.addRef());
                final CudaMemory backwardsFilterWorkSpace = gpu.allocateBackwardFilterWorkspace(
                    inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
                    convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(), backwardFilterAlgorithm,
                    1);
                @Nonnull
                CudaMemory filterPtr = gpu.allocate((long) kernelLength * precision.size, MemoryType.Device,
                    true);
                try {
                  CudaMemory inputTensorMemory = inputTensor.getMemory(gpu.addRef());
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu.addRef(), MemoryType.Managed.ifEnabled());
                  //              inputTensorMemory.synchronize();
                  assert deltaTensorMemory != null;
                  assert inputTensorMemory != null;
                  CudaSystem.handle(gpu.cudnnConvolutionBackwardFilter(precision.getPointer(1.0),
                      inputTensor.descriptor.getPtr(), inputTensorMemory.getPtr(),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                      convolutionDescriptor.getPtr(), backwardFilterAlgorithm,
                      backwardsFilterWorkSpace.getPtr(), backwardsFilterWorkSpace.size,
                      precision.getPointer(0.0), filterDescriptor.getPtr(), filterPtr.getPtr()));
                  gpu.freeRef();
                  filterPtr.dirty();
                  deltaTensorMemory.dirty();
                  deltaTensorMemory.freeRef();
                  inputTensorMemory.dirty();
                  inputTensorMemory.freeRef();
                  backwardsFilterWorkSpace.dirty();
                  backwardsFilterWorkSpace.freeRef();
                  deltaTensor.freeRef();
                  inputTensor.freeRef();
                  filterDescriptor.freeRef();
                  convolutionDescriptor.freeRef();
                  Tensor temp_16_0007 = filterPtr.read(precision, kernelDimensions);
                  filterPtr.freeRef();
                  return temp_16_0007;
                } catch (@Nonnull final Throwable e) {
                  throw new ComponentException(
                      RefString.format("Error in convolution %s x %s => %s", RefArrays.toString(inputDims),
                          RefArrays.toString(kernelDimensions), RefArrays.toString(outputSize)),
                      e);
                }
              }, inputData.addRef(), delta.addRef()), delta.addRef());
          Delta<UUID> temp_16_0020 = buffer.get(id, kernelData);
          assert temp_16_0020 != null;
          temp_16_0020.addInPlace(weightGradient.getData());
          temp_16_0020.freeRef();
          weightGradient.freeRef();
          simpleConvolutionLayer.clearCudaFilters();
        }
      }, inputData.addRef(), delta.addRef(), buffer.addRef());
      Runnable backpropFn = RefUtil.wrapInterface(() -> {
        if (alive) {
          final TensorList inputBufferTensors = CudaSystem
              .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                return simpleConvolutionLayer.bck_workaround(gpu, paddingX, paddingY, inputLength,
                    inputDims, kernelDimensions, outputSize, delta.addRef());
              }, delta.addRef()), delta.addRef());
          if (null != inputBufferTensors) {
            inputAccumulator.accept(buffer.addRef(), inputBufferTensors);
          }
        }
      }, delta, inputAccumulator.addRef(), buffer);
      if (CoreSettings.INSTANCE().isSingleThreaded()) {
        Util.runAllSerial(learnFn, backpropFn);
      } else {
        Util.runAllParallel(learnFn, backpropFn);
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      inputData.freeRef();
      inputAccumulator.freeRef();
      simpleConvolutionLayer.freeRef();
    }
  }
}
