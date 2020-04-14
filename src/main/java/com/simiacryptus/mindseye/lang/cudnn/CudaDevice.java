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

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.ref.wrappers.*;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.charset.Charset;

/**
 * The type Cuda device.
 */
public class CudaDevice extends CudaSystem {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudnnHandle.class);
  /**
   * The Device name.
   */
  @Nullable
  protected final String deviceName;
  /**
   * The Device id.
   */
  protected final int deviceId;
  /**
   * The Allocation lock.
   */
  final Object allocationLock = new Object();
  private final Object memoryManagementLock = new Object();
  private volatile cudaDeviceProp deviceProperties;

  /**
   * Instantiates a new Cuda device.
   *
   * @param deviceId the device id
   */
  protected CudaDevice(final int deviceId) {
    super();
    this.deviceId = deviceId;
    assert 0 <= this.deviceId;
    initThread();
    deviceName = getDeviceName(deviceId);
  }

  /**
   * Gets device id.
   *
   * @return the device id
   */
  public int getDeviceId() {
    return deviceId;
  }

  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(final int cudaDeviceId) {
    if (cudaDeviceId < 0)
      throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
    if (!isThreadDeviceId(cudaDeviceId)) {
      long startTime = RefSystem.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
      log("cudaSetDevice", result, new Object[]{cudaDeviceId});
      handle(result);
      currentDeviceId.set(cudaDeviceId);
    }
  }

  /**
   * Cuda free.
   *
   * @param deviceId the device id
   * @param devPtr   the dev ptr
   */
  public static void cudaFree(int deviceId, @Nullable final CudaPointer devPtr) {
    long startTime = RefSystem.nanoTime();
    if (null == devPtr)
      return;
    RefConsumer<CudnnHandle> fn = dev -> {
      final int result = JCuda.cudaFree(devPtr);
      log("cudaFree", result, new Object[]{devPtr});
      cudaFree_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
      handle(result);
      dev.freeRef();
    };
    if (deviceId < 0) {
      fn.accept(null);
    } else {
      withDevice(deviceId, fn);
    }
  }

  /**
   * Gets device name.
   *
   * @param device the device
   * @return the device name
   */
  @Nonnull
  public static String getDeviceName(final int device) {
    return new String(CudaDevice.getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }

  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  @NotNull
  public static cudaDeviceProp getDeviceProperties(final int device) {
    return propertyCache.computeIfAbsent(device, deviceId -> {
      long startTime = RefSystem.nanoTime();
      @Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
      log("cudaGetDeviceProperties", result, new Object[]{deviceProp, device});
      return deviceProp;
    });
  }

  /**
   * Ensure capacity device metrics.
   *
   * @param size the size
   * @return the device metrics
   */
  @Nonnull
  public DeviceMetrics ensureCapacity(final long size) {
    if (size <= 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > (double) CudaSettings.INSTANCE().maxAllocSize) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    final DeviceMetrics metrics;
    synchronized (memoryManagementLock) {
      metrics = CudaMemory.getGpuStats(deviceId);
      RefCollection<DeviceMetrics> temp_75_0001 = CudaMemory.METRICS.values();
      double resultingTotalMemory = temp_75_0001.stream().mapToLong(m -> m.usedMemory.get()).sum() + size;
      temp_75_0001.freeRef();
      if (resultingTotalMemory > (double) CudaSettings.INSTANCE().maxTotalMemory) {
        CudaMemory.logger.info(RefString.format("Clearing weak global memory while allocating %e bytes (%e > %e)",
            (double) size, resultingTotalMemory, (double) CudaSettings.INSTANCE().maxTotalMemory));
        CudaMemory.clearWeakMemory(deviceId);
      }
      RefCollection<DeviceMetrics> temp_75_0002 = CudaMemory.METRICS.values();
      resultingTotalMemory = temp_75_0002.stream().mapToLong(x1 -> x1.usedMemory.get()).sum() + size;
      temp_75_0002.freeRef();
      if (resultingTotalMemory > (double) CudaSettings.INSTANCE().maxTotalMemory) {
        CudaMemory.logger.info(RefString.format("Clearing all global memory while allocating %e bytes (%e > %e)",
            (double) size, resultingTotalMemory, (double) CudaSettings.INSTANCE().maxTotalMemory));
        CudaMemory.clearMemory(deviceId);
      }
      double resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > (double) CudaSettings.INSTANCE().maxDeviceMemory) {
        CudaMemory.logger
            .info(RefString.format("Clearing weak memory for device %s while allocating %e bytes (%e > %e)", this,
                (double) size, resultingDeviceMemory, (double) CudaSettings.INSTANCE().maxDeviceMemory));
        RefSet<Integer> temp_75_0003 = CudaMemory.METRICS.keySet();
        temp_75_0003.stream().mapToInt(x -> x).distinct().forEach(deviceId1 -> CudaMemory.clearWeakMemory(deviceId1));
        temp_75_0003.freeRef();
      }
      resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > (double) CudaSettings.INSTANCE().maxDeviceMemory) {
        CudaMemory.logger.info(RefString.format("Clearing all memory for device %s while allocating %e bytes (%s > %e)",
            this, (double) size, resultingDeviceMemory, (double) CudaSettings.INSTANCE().maxDeviceMemory));
        RefSet<Integer> temp_75_0004 = CudaMemory.METRICS.keySet();
        temp_75_0004.stream().mapToInt(x -> x).distinct().forEach(deviceId1 -> CudaMemory.clearMemory(deviceId1));
        temp_75_0004.freeRef();
      }
    }
    return metrics;
  }

  /**
   * Allocate cuda memory.
   *
   * @param size  the size
   * @param type  the type
   * @param dirty the dirty
   * @return the cuda memory
   */
  @Nonnull
  public CudaMemory allocate(final long size, @Nonnull MemoryType type, boolean dirty) {
    assert isThreadDeviceId(getDeviceId());
    @Nonnull
    CudaMemory obtain = new CudaMemory(addRef(), size, type);
    if (!dirty)
      obtain.clear();
    return obtain;
  }

  /**
   * New tensor descriptor cuda tensor descriptor.
   *
   * @param dataType   the data type
   * @param batchCount the batch count
   * @param channels   the channels
   * @param height     the height
   * @param width      the width
   * @return the cuda tensor descriptor
   */
  @Nonnull
  public CudaTensorDescriptor newTensorDescriptor(@Nonnull final Precision dataType, final int batchCount, final int channels,
                                                  final int height, final int width) {
    return newTensorDescriptor(dataType, batchCount, channels, height, width, channels * height * width, height * width,
        width, 1);
  }

  /**
   * New tensor descriptor cuda tensor descriptor.
   *
   * @param dataType   the data type
   * @param batchCount the batch count
   * @param channels   the channels
   * @param height     the height
   * @param width      the width
   * @param nStride    the n stride
   * @param cStride    the c stride
   * @param hStride    the h stride
   * @param wStride    the w stride
   * @return the cuda tensor descriptor
   */
  @Nonnull
  public CudaTensorDescriptor newTensorDescriptor(@Nonnull final Precision dataType, final int batchCount, final int channels,
                                                  final int height, final int width, final int nStride, final int cStride, final int hStride, final int wStride) {
    assert batchCount > 0;
    assert channels > 0;
    assert height > 0;
    assert width > 0;
    assert nStride > 0;
    assert cStride > 0;
    assert hStride != 0;
    assert wStride != 0;
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    log("cudnnCreateTensorDescriptor", result, new Object[]{desc});
    handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType.code, batchCount, channels, height, width, nStride,
        cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetTensor4dDescriptorEx", result,
        new Object[]{desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride});
    handle(result);
    return new CudaTensorDescriptor(desc, getDeviceId(), dataType, batchCount, channels, height, width, nStride,
        cStride, hStride, wStride);
  }

  /**
   * New op descriptor cuda resource.
   *
   * @param opType   the op type
   * @param dataType the data type
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(final int opType, @Nonnull final Precision dataType) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    log("cudnnCreateOpTensorDescriptor", result, new Object[]{opDesc});
    handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType.code,
        cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetOpTensorDescriptor", result,
        new Object[]{opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN});
    handle(result);
    return new CudaResource<>(opDesc, opTensorDesc -> CudaSystem.cudnnDestroyOpTensorDescriptor(opTensorDesc), getDeviceId());
  }

  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType       the data type
   * @param tensorLayout   the tensor layout
   * @param outputChannels the output channels
   * @param inputChannels  the input channels
   * @param height         the height
   * @param width          the width
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnFilterDescriptor> newFilterDescriptor(@Nonnull final Precision dataType, final int tensorLayout,
                                                                 final int outputChannels, final int inputChannels, final int height, final int width) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, new Object[]{filterDesc});
    handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType.code, tensorLayout, outputChannels, inputChannels,
        height, width);
    newFilterDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetFilter4dDescriptor", result,
        new Object[]{filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width});
    handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, filterDesc1 -> CudaSystem.cudnnDestroyFilterDescriptor(filterDesc1),
        getDeviceId()) {
      {
      }

      @Nonnull
      @Override
      public String toString() {
        return "cudnnSetFilter4dDescriptor(dataType=" + dataType + ";tensorLayout=" + tensorLayout + ";outputChannels="
            + outputChannels + ";inputChannels=" + inputChannels + ";height=" + height + ";=width" + width + ")";
      }

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }
    };
  }

  /**
   * New convolutions 2 d descriptor cuda resource.
   *
   * @param mode         the mode
   * @param dataType     the data type
   * @param paddingY     the padding y
   * @param paddingX     the padding x
   * @param strideHeight the stride height
   * @param strideWidth  the stride width
   * @param dilationY    the dilation y
   * @param dilationX    the dilation x
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int mode, @Nonnull final Precision dataType,
                                                                              final int paddingY, final int paddingX, final int strideHeight, final int strideWidth, int dilationY,
                                                                              int dilationX) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    log("cudnnCreateConvolutionDescriptor", result, new Object[]{convDesc});
    handle(result);
    result = JCudnn.cudnnSetConvolution2dDescriptor(convDesc, paddingY, // zero-padding height
        paddingX, // zero-padding width
        strideHeight, // vertical filter stride
        strideWidth, // horizontal filter stride
        dilationY, // upscale the input in x-direction
        dilationX, // upscale the input in y-direction
        mode, dataType.code);
    newConvolutions2dDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetConvolution2dDescriptor", result,
        new Object[]{convDesc, paddingY, paddingX, strideHeight, strideWidth, dilationY, dilationX, mode, dataType});
    handle(result);
    return new CudaResource<>(convDesc, convDesc1 -> CudaSystem.cudnnDestroyConvolutionDescriptor(convDesc1), getDeviceId());
  }

  /**
   * New activation descriptor cuda resource.
   *
   * @param mode     the mode
   * @param reluNan  the relu nan
   * @param reluCeil the relu ceil
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnActivationDescriptor> newActivationDescriptor(final int mode, final int reluNan,
                                                                         final double reluCeil) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    log("cudnnCreateActivationDescriptor", result, new Object[]{desc});
    handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetActivationDescriptor", result, new Object[]{desc, mode, reluNan, reluCeil});
    handle(result);
    return new CudaResource<>(desc, activationDesc -> CudaSystem.cudnnDestroyActivationDescriptor(activationDesc), getDeviceId());
  }

  /**
   * Create pooling descriptor cuda resource.
   *
   * @param mode       the mode
   * @param poolDims   the pool dims
   * @param windowSize the window size
   * @param padding    the padding
   * @param stride     the stride
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(final int mode, final int poolDims,
                                                                      final int[] windowSize, final int[] padding, final int[] stride) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    log("cudnnCreatePoolingDescriptor", result, new Object[]{poolingDesc});
    handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc, mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        poolDims, windowSize, padding, stride);
    log("cudnnSetPoolingNdDescriptor", result, new Object[]{poolingDesc, mode,
        cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize, padding, stride});
    handle(result);
    createPoolingDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, poolingDesc1 -> CudaSystem.cudnnDestroyPoolingDescriptor(poolingDesc1), getDeviceId());
  }

  /**
   * Create lrn descriptor cuda resource.
   *
   * @param lrnN     the lrn n
   * @param lrnAlpha the lrn alpha
   * @param lrnBeta  the lrn beta
   * @param lrnK     the lrn k
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnLRNDescriptor> createLRNDescriptor(int lrnN, double lrnAlpha, double lrnBeta, double lrnK) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final cudnnLRNDescriptor poolingDesc = new cudnnLRNDescriptor();
    int result = JCudnn.cudnnCreateLRNDescriptor(poolingDesc);
    log("cudnnCreateLRNDescriptor", result, new Object[]{poolingDesc});
    handle(result);
    result = JCudnn.cudnnSetLRNDescriptor(poolingDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
    log("cudnnSetLRNDescriptor", result, new Object[]{poolingDesc, lrnN, lrnAlpha, lrnBeta, lrnK});
    handle(result);
    createLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, lrnDesc -> JCudnn.cudnnDestroyLRNDescriptor(lrnDesc), getDeviceId());
  }

  /**
   * Init thread.
   */
  public void initThread() {
    setDevice(getDeviceId());
  }

  @Nonnull
  @Override
  public CudaDevice addRef() {
    return (CudaDevice) super.addRef();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  /**
   * Acquire cuda pointer.
   *
   * @param size    the size
   * @param type    the type
   * @param retries the retries
   * @return the cuda pointer
   */
  @Nonnull
  CudaPointer acquire(long size, @Nonnull MemoryType type, int retries) {
    if (size <= 0)
      throw new IllegalArgumentException();
    assert isThreadDeviceId(getDeviceId());
    if (retries < 0)
      throw new IllegalArgumentException();
    return _acquire(size, type, retries);
  }

  @Nonnull
  private CudaPointer _acquire(long size, @Nonnull MemoryType type, int retries) {
    synchronized (allocationLock) {
      final DeviceMetrics metrics = ensureCapacity(size);
      @Nonnull
      CudaPointer pointer = null;
      try {
        pointer = type.allocCached(size, this.addRef());
        final long finalMemory = metrics.activeMemory.addAndGet(size);
        metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      } catch (@Nonnull final ThreadDeath e) {
        throw e;
      } catch (@Nonnull final Throwable e) {
        if (retries <= 0)
          throw new RuntimeException(
              RefString.format(RefString.format("Error allocating %e bytes; %s currently allocated to device %s",
                  (double) size, metrics.usedMemory, this.addRef())),
              e);
        final long startMemory = metrics.usedMemory.get();
        @Nonnull
        TimedResult<Double> timedResult = TimedResult.time(() -> CudaMemory.clearMemory(getDeviceId()));
        final long freedMemory = startMemory - metrics.usedMemory.get();
        CudaMemory.logger.warn(RefString.format(
            "Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)", size,
            freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
        timedResult.freeRef();
      }
      assert pointer != null;
      return pointer;
    }
  }

  /**
   * The type Cuda tensor descriptor.
   */
  public static class CudaTensorDescriptor extends CudaResource<cudnnTensorDescriptor> {

    /**
     * The W stride.
     */
    public final int wStride;
    /**
     * The H stride.
     */
    public final int hStride;
    /**
     * The C stride.
     */
    public final int cStride;
    /**
     * The N stride.
     */
    public final int nStride;
    /**
     * The Width.
     */
    public final int width;
    /**
     * The Height.
     */
    public final int height;
    /**
     * The Channels.
     */
    public final int channels;
    /**
     * The Batch count.
     */
    public final int batchCount;
    /**
     * The Data type.
     */
    public final Precision dataType;

    /**
     * Instantiates a new Cuda tensor descriptor.
     *
     * @param obj        the obj
     * @param deviceId   the device id
     * @param dataType   the data type
     * @param batchCount the batch count
     * @param channels   the channels
     * @param height     the height
     * @param width      the width
     * @param nStride    the n stride
     * @param cStride    the c stride
     * @param hStride    the h stride
     * @param wStride    the w stride
     */
    protected CudaTensorDescriptor(final cudnnTensorDescriptor obj, final int deviceId, final Precision dataType,
                                   final int batchCount, final int channels, final int height, final int width, final int nStride,
                                   final int cStride, final int hStride, final int wStride) {
      super(obj, tensorDesc -> CudaSystem.cudnnDestroyTensorDescriptor(tensorDesc), deviceId);
      this.dataType = dataType;
      this.batchCount = batchCount;
      this.channels = channels;
      this.height = height;
      this.width = width;
      this.nStride = nStride;
      this.cStride = cStride;
      this.hStride = hStride;
      this.wStride = wStride;
    }

    /**
     * Copy cuda tensor descriptor.
     *
     * @param device the device
     * @return the cuda tensor descriptor
     */
    @Nonnull
    public CudaTensorDescriptor copy(@Nonnull CudaDevice device) {
      CudaTensorDescriptor tensorDescriptor = device.newTensorDescriptor(
          dataType, batchCount, channels, height, width,
          nStride, cStride, hStride, wStride);
      device.freeRef();
      return tensorDescriptor;
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    CudaTensorDescriptor addRef() {
      return (CudaTensorDescriptor) super.addRef();
    }
  }
}
