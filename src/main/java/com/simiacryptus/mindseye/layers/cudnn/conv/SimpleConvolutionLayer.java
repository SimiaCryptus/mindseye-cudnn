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

/**
 * The type Simple convolution layer.
 */
@SuppressWarnings("serial")
public class SimpleConvolutionLayer extends LayerBase implements MultiPrecision {

  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(SimpleConvolutionLayer.class);
  @Nonnull
  private final Tensor kernel;
  @Nullable
  private final RefMap<Integer, CudaMemory> gpuFilters = new RefConcurrentHashMap<>();
  private int paddingX;
  private int paddingY;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private int strideX = 1;
  private int strideY = 1;

  /**
   * Instantiates a new Simple convolution layer.
   */
  protected SimpleConvolutionLayer() {
    this(null);
  }

  /**
   * Instantiates a new Simple convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public SimpleConvolutionLayer(final int width, final int height, final int bands) {
    this(new Tensor(width, height, bands));
  }

  /**
   * Instantiates a new Simple convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
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

  /**
   * Instantiates a new Simple convolution layer.
   *
   * @param kernel the kernel
   */
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

  /**
   * Instantiates a new Simple convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
  public SimpleConvolutionLayer(int width, int height, int inputBands, int outputBands) {
    this(width, height, inputBands * outputBands);
  }

  /**
   * Gets kernel.
   *
   * @return the kernel
   */
  @Nullable
  public Tensor getKernel() {
    return kernel.addRef();
  }

  /**
   * Get kernel dimensions int [ ].
   *
   * @return the int [ ]
   */
  @Nonnull
  public int[] getKernelDimensions() {
    return kernel.getDimensions();
  }

  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }

  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   */
  public void setPaddingX(int paddingX) {
    this.paddingX = paddingX;
  }

  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }

  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   */
  public void setPaddingY(int paddingY) {
    this.paddingY = paddingY;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    gpuFilters.clear();
    this.precision = precision;
  }

  /**
   * Gets stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }

  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   */
  public void setStrideX(int strideX) {
    this.strideX = strideX;
  }

  /**
   * Gets stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }

  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   */
  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }

  /**
   * Sets weights log.
   *
   * @param f the f
   */
  public void setWeightsLog(double f) {
    set(() -> Math.pow(10, f) * (Math.random() - 0.5));
  }

  /**
   * From json simple convolution layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the simple convolution layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SimpleConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SimpleConvolutionLayer(json, rs);
  }

  /**
   * Reverse int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  @Nonnull
  public static int[] reverse(@Nonnull int... array) {
    for (int i = 0; i < array.length / 2; i++) {
      int j = array[array.length - (i + 1)];
      array[array.length - (i + 1)] = array[i];
      array[i] = j;
    }
    return array;
  }

  /**
   * Bck cuda tensor list.
   *
   * @param precision        the precision
   * @param strideX          the stride x
   * @param strideY          the stride y
   * @param gpu              the gpu
   * @param paddingX         the padding x
   * @param paddingY         the padding y
   * @param inputLength      the input length
   * @param inputDims        the input dims
   * @param kernelDimensions the kernel dimensions
   * @param outputSize       the output size
   * @param delta            the delta
   * @param filterPtr        the filter ptr
   * @return the cuda tensor list
   */
  @Nullable
  public static CudaTensorList bck(Precision precision, int strideX, int strideY, @Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, @Nonnull int[] inputDims,
                                   int[] kernelDimensions, int[] outputSize, @Nullable TensorList delta, @Nonnull CudaMemory filterPtr) {
    final CudaDevice.CudaTensorDescriptor inputDescriptor = newTensorDescriptor(precision, gpu.addRef(), inputLength, inputDims);
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
      return new CudaTensorList(new CudaTensor(passbackMemory,
          inputDescriptor.addRef(), precision), inputLength, inputDims, precision);
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException(RefString.format("Error in convolution %s x %s => %s", RefArrays.toString(inputDims),
          RefArrays.toString(kernelDimensions), RefArrays.toString(outputSize)), e);
    } finally {
      inputDescriptor.freeRef();
      filterDescriptor.freeRef();
      convolutionDescriptor.freeRef();
      deltaTensor.freeRef();
      backwardsDataWorkSpace.freeRef();
      filterPtr.freeRef();
      gpu.freeRef();
    }
  }

  /**
   * Fwd cuda tensor list.
   *
   * @param precision        the precision
   * @param strideX          the stride x
   * @param strideY          the stride y
   * @param gpu              the gpu
   * @param paddingX         the padding x
   * @param paddingY         the padding y
   * @param inputLength      the input length
   * @param inputDims        the input dims
   * @param kernelDimensions the kernel dimensions
   * @param outputSize       the output size
   * @param inputData        the input data
   * @param filterPtr        the filter ptr
   * @return the cuda tensor list
   */
  @Nullable
  public static CudaTensorList fwd(Precision precision, int strideX, int strideY, @Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, int[] inputDims,
                                   int[] kernelDimensions, int[] outputSize, @Nullable TensorList inputData, @Nonnull CudaMemory filterPtr) {
    CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(
        cudnnConvolutionMode.CUDNN_CONVOLUTION, precision, paddingY, paddingX, strideY, strideX, 1, 1);
    CudaTensor tensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
    CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(precision,
        cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputDims[2], kernelDimensions[1], kernelDimensions[0]);
    String msg = RefString.format("Error in convolution %s x %s", RefArrays.toString(inputDims),
        RefArrays.toString(kernelDimensions));
    final CudaDevice.CudaTensorDescriptor inputDescriptor = newTensorDescriptor(precision, gpu.addRef(), inputLength, inputDims);
    final int[] outputDims = RefIntStream.of(reverse(
        CudaSystem.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr())))
        .limit(3).toArray();
    final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(precision, gpu.addRef(), inputLength, outputDims);
    final int forwardAlgorithm = SimpleConvolutionLayer.getForwardAlgorithm(gpu.addRef(), tensor.addRef(),
        filterDescriptor.addRef(), convolutionDescriptor.addRef(), outputDescriptor.addRef());
    final CudaMemory forwardWorkspace = gpu.allocateForwardWorkspace(inputDescriptor.getPtr(),
        filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), forwardAlgorithm, 1);
    try {
      @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) Tensor.length(outputDims) * inputLength * precision.size,
          MemoryType.Managed.ifEnabled(), true);
      CudaMemory inputTensorMemory = tensor.getMemory(gpu.addRef());
      //        inputTensorMemory.synchronize();
      assert inputTensorMemory != null;
      CudaSystem.handle(gpu.cudnnConvolutionForward(
          precision.getPointer(1.0),
          inputDescriptor.getPtr(),
          inputTensorMemory.getPtr(),
          filterDescriptor.getPtr(),
          filterPtr.getPtr(),
          convolutionDescriptor.getPtr(),
          forwardAlgorithm,
          forwardWorkspace.getPtr(),
          forwardWorkspace.size,
          precision.getPointer(0.0),
          outputDescriptor.getPtr(),
          outputBuffer.getPtr()
      ));
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

  /**
   * Gets forward algorithm.
   *
   * @param gpu                   the gpu
   * @param inputTensor           the input tensor
   * @param filterDescriptor      the filter descriptor
   * @param convolutionDescriptor the convolution descriptor
   * @param outputDescriptor      the output descriptor
   * @return the forward algorithm
   */
  public static int getForwardAlgorithm(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor inputTensor,
                                        @Nonnull final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                        @Nonnull final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                        final CudaDevice.CudaTensorDescriptor outputDescriptor) {
    int gpuForwardAlgorithm = gpu.getForwardAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), outputDescriptor.getPtr(),
        CudaSettings.INSTANCE().convolutionWorkspaceSizeLimit);
    gpu.freeRef();
    outputDescriptor.freeRef();
    convolutionDescriptor.freeRef();
    filterDescriptor.freeRef();
    inputTensor.freeRef();
    //    return cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    return gpuForwardAlgorithm;
  }

  /**
   * Gets backward filter algorithm.
   *
   * @param gpu                   the gpu
   * @param deltaTensor           the delta tensor
   * @param inputTensor           the input tensor
   * @param filterDescriptor      the filter descriptor
   * @param convolutionDescriptor the convolution descriptor
   * @return the backward filter algorithm
   */
  public static int getBackwardFilterAlgorithm(@Nonnull final CudnnHandle gpu, @Nonnull final CudaTensor deltaTensor,
                                               @Nonnull final CudaTensor inputTensor, @Nonnull final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                               @Nonnull final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    int backwardFilterAlgorithm = gpu.getBackwardFilterAlgorithm(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(),
        CudaSettings.INSTANCE().convolutionWorkspaceSizeLimit);
    convolutionDescriptor.freeRef();
    filterDescriptor.freeRef();
    inputTensor.freeRef();
    deltaTensor.freeRef();
    gpu.freeRef();
    return backwardFilterAlgorithm;
  }

  /**
   * Gets backward data algorithm.
   *
   * @param gpu                   the gpu
   * @param dyDescriptor          the dy descriptor
   * @param filterDescriptor      the filter descriptor
   * @param convolutionDescriptor the convolution descriptor
   * @param dxDescriptor          the dx descriptor
   * @return the backward data algorithm
   */
  public static int getBackwardDataAlgorithm(@Nonnull final CudnnHandle gpu, final CudaDevice.CudaTensorDescriptor dyDescriptor,
                                             @Nullable final CudaResource<cudnnFilterDescriptor> filterDescriptor,
                                             @Nullable final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor,
                                             CudaDevice.CudaTensorDescriptor dxDescriptor) {
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

  private static CudaTensorList bck2(Tensor kernel, Precision precision, int strideX, int strideY, @Nonnull CudnnHandle gpu, int paddingX, int paddingY, int inputLength, @Nonnull int[] inputDims, @Nonnull int[] outputSize, @Nullable TensorList delta) {
    CudaMemory cudaMemory = gpu.allocate((long) kernel.length() * precision.size, MemoryType.Device, true);
    Tensor reshapeCast = kernel.reshapeCast(1, 1, inputDims[2], outputSize[2]);
    kernel.freeRef();
    cudaMemory.write(precision, reshapeCast.permuteDimensions(0, 1, 3, 2));
    reshapeCast.freeRef();
    return fwd(precision, strideX, strideY, gpu, paddingX, paddingY, inputLength, outputSize, new int[]{1, 1, outputSize[2], inputDims[2]}, inputDims, delta, cudaMemory);
  }

  private static CudaDevice.CudaTensorDescriptor newTensorDescriptor(Precision precision, @Nonnull CudnnHandle gpu, int inputLength, int[] outputDims) {
    CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, inputLength,
        outputDims[2], outputDims[1], outputDims[0], outputDims[2] * outputDims[1] * outputDims[0],
        outputDims[1] * outputDims[0], outputDims[0], 1);
    gpu.freeRef();
    return tensorDescriptor;
  }

  @Nonnull
  private static CudaMemory getCudaFilter(RefMap<Integer, CudaMemory> gpuFilters, Tensor kernel, Precision precision, @Nonnull final CudaDevice gpu) {
    if (CudaSettings.INSTANCE().convolutionCache) return getCudaFilter_cached(gpuFilters, kernel, precision, gpu);
    else {
      gpuFilters.freeRef();
      return getCudaFilter_instance(kernel, precision, gpu);
    }
  }

  @Nonnull
  private static CudaMemory getCudaFilter_instance(Tensor kernel, Precision precision, @Nonnull final CudaDevice gpu) {
    CudaMemory cudaMemory = gpu.allocate((long) kernel.length() * precision.size, MemoryType.Device, true);
    if (!(kernel.rms() > 0)) {
      log.warn("No data in kernel");
      //throw new AssertionError();
    }
    cudaMemory.write(precision, kernel);
    gpu.freeRef();
    return cudaMemory;
  }

  @Nonnull
  @RefIgnore
  private static CudaMemory getCudaFilter_cached(RefMap<Integer, CudaMemory> gpuFilters, Tensor kernel, Precision precision, @Nonnull final CudaDevice device) {
    assert gpuFilters != null;
    try {
      synchronized (gpuFilters) {
        return gpuFilters.computeIfAbsent(device.getDeviceId(), deviceId -> {
          return getCudaFilter_instance(kernel.addRef(), precision, device);
        });
      }
    } finally {
      kernel.freeRef();
      gpuFilters.freeRef();
    }
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assertAlive();
    TensorList data0 = inObj[0].getData();
    @Nonnull final int[] rawInputDims = data0.getDimensions();
    data0.freeRef();
    final int[] kernelDimensions = getKernelDimensions();
    int kernelLength = kernel.length();
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
    @Nullable final Result input;
    if (correctionX > 0 || correctionY > 0) {
      ImgCropLayer imgCropLayer = new ImgCropLayer(rawInputDims[0] + correctionX, rawInputDims[1] + correctionY);
      imgCropLayer.setPrecision(precision);
      imgCropLayer.setHorizontalAlign(ImgCropLayer.Alignment.Center);
      imgCropLayer.setVerticalAlign(ImgCropLayer.Alignment.Center);
      imgCropLayer.setRoundUp(false);
      input = imgCropLayer.eval(inObj[0].addRef());
      imgCropLayer.freeRef();
    } else {
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
        precision,
        strideX,
        strideY,
        inputData,
        inputLength,
        outputSize,
        inputDims,
        kernelDimensions,
        paddingY,
        paddingX,
        kernelLength,
        kernel.addRef(),
        inputAlive,
        input.getAccumulator(),
        getId(),
        isFrozen(),
        gpuFilters.addRef()
    );
    input.freeRef();
    return new Result(data, accumulator, inputAlive || !isFrozen());
  }

  /**
   * Evict device data long.
   *
   * @param deviceId the device id
   * @return the long
   */
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
      throw new RuntimeException("Error serializing convolution" + RefArrays.toString(this.getKernelDimensions()), e);
    }
    json.add("filter", value);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("precision", precision.name());
    return json;
  }

  /**
   * Get output size int [ ].
   *
   * @param inputSize the input size
   * @return the int [ ]
   */
  public int[] getOutputSize(final int... inputSize) {
    @Nonnull final int[] kernelSize = getKernelDimensions();
    if (kernelSize.length > inputSize.length) {
      throw new IllegalArgumentException();
    }
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

  /**
   * Set.
   *
   * @param f the f
   */
  public void set(@Nonnull DoubleSupplier f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
    assert kernel.rms() > 0;
    gpuFilters.clear();
  }

  /**
   * Set.
   *
   * @param f the f
   */
  public void set(@Nonnull ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
    assert kernel.rms() > 0;
    gpuFilters.clear();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  /**
   * Sets padding xy.
   *
   * @param x the x
   * @param y the y
   */
  public void setPaddingXY(int x, int y) {
    setPaddingX(x);
    setPaddingY(y);
  }

  /**
   * Set.
   *
   * @param kernel the kernel
   */
  public void set(@Nonnull Tensor kernel) {
    assert kernel.rms() > 0;
    this.kernel.set(kernel);
    assert this.kernel.rms() > 0;
    gpuFilters.clear();
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

  /**
   * Sets stride xy.
   *
   * @param x the x
   * @param y the y
   */
  public void setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
  }

  public void _free() {
    super._free();
    gpuFilters.clear();
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
    return CudaSystem.run(gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert 0 < kernelLength;
      assert kernelDimensions[0] * kernelDimensions[1] * kernelDimensions[2] == kernelLength;
      CudaMemory cudaFilter = getCudaFilter(gpuFilters.addRef(), kernel.addRef(), precision, gpu.addRef());
      return fwd(precision, strideX, strideY, gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, inputData.addRef(), cudaFilter);
    }, inputData);
  }

  private int correct(int dim, int modulus, int offset) {
    int adj = modulus - (dim - offset) % modulus;
    while (adj < 0)
      adj += modulus;
    while (adj >= modulus)
      adj -= modulus;
    return adj;
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final int inputLength;
    private final int[] outputSize;
    private final int[] inputDims;
    private final int[] kernelDimensions;
    private final int paddingY;
    private final int paddingX;
    private final int kernelLength;
    private final Tensor kernel;
    private final boolean alive;
    private UUID id;
    private Result.Accumulator inputAccumulator;
    private Precision precision;
    private int strideX;
    private int strideY;
    private boolean frozen;
    private RefMap<Integer, CudaMemory> gpuFilters;

    /**
     * Instantiates a new Accumulator.
     *
     * @param precision        the precision
     * @param strideX          the stride x
     * @param strideY          the stride y
     * @param inputData        the input data
     * @param inputLength      the input length
     * @param outputSize       the output size
     * @param inputDims        the input dims
     * @param kernelDimensions the kernel dimensions
     * @param paddingY         the padding y
     * @param paddingX         the padding x
     * @param kernelLength     the kernel length
     * @param kernel           the kernel
     * @param alive            the alive
     * @param accumulator      the accumulator
     * @param id               the id
     * @param frozen           the frozen
     * @param gpuFilters       the gpu filters
     */
    public Accumulator(Precision precision, int strideX, int strideY, TensorList inputData, int inputLength, int[] outputSize, int[] inputDims, int[] kernelDimensions, int paddingY, int paddingX, int kernelLength, Tensor kernel, boolean alive, Result.Accumulator accumulator, UUID id, boolean frozen, RefMap<Integer, CudaMemory> gpuFilters) {
      this.inputData = inputData;
      this.inputLength = inputLength;
      this.outputSize = outputSize;
      this.inputDims = inputDims;
      this.kernelDimensions = kernelDimensions;
      this.paddingY = paddingY;
      this.paddingX = paddingX;
      this.kernelLength = kernelLength;
      this.kernel = kernel;
      this.alive = alive;
      this.id = id;
      inputAccumulator = accumulator;
      this.precision = precision;
      this.strideX = strideX;
      this.strideY = strideY;
      this.frozen = frozen;
      this.gpuFilters = gpuFilters;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      delta.assertAlive();
      buffer.assertAlive();
      inputData.assertAlive();
      assert delta.length() == inputLength;
      Runnable learnFn = RefUtil.wrapInterface(() -> {
        if (!frozen) {
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
                final int backwardFilterAlgorithm = SimpleConvolutionLayer.getBackwardFilterAlgorithm(
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
          Delta<UUID> kernelDelta = buffer.get(id, kernel.addRef());
          assert kernelDelta != null;
          kernelDelta.addInPlace(weightGradient);
          kernelDelta.freeRef();
          gpuFilters.clear();
        }
      }, inputData.addRef(), delta.addRef(), buffer.addRef());
      Runnable backpropFn = RefUtil.wrapInterface(() -> {
        if (alive) {
          final TensorList inputBufferTensors = CudaSystem
              .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                TensorList delta1 = delta.addRef();
                if (1 != kernelDimensions[0] || 1 != kernelDimensions[1]) {
                  CudaMemory cudaFilter = SimpleConvolutionLayer.getCudaFilter(gpuFilters.addRef(), kernel.addRef(), precision, gpu.addRef());
                  return bck(precision, strideX, strideY, gpu, paddingX, paddingY, inputLength, inputDims, kernelDimensions, outputSize, delta1, cudaFilter);
                } else {
                  return bck2(kernel.addRef(), precision, strideX, strideY, gpu, paddingX, paddingY, inputLength, inputDims, outputSize, delta1);
                }
              }, delta.addRef()), delta.addRef());
          if (null != inputBufferTensors) {
            inputAccumulator.accept(buffer.addRef(), inputBufferTensors);
          }
        }
      }, delta, inputAccumulator.addRef(), buffer);
      if (CoreSettings.INSTANCE().singleThreaded) {
        Util.runAllSerial(learnFn, backpropFn);
      } else {
        Util.runAllParallel(learnFn, backpropFn);
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      kernel.freeRef();
      inputData.freeRef();
      inputAccumulator.freeRef();
      gpuFilters.freeRef();
    }
  }
}
