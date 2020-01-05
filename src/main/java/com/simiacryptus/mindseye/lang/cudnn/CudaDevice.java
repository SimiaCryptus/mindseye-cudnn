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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefCollection;
import com.simiacryptus.ref.wrappers.RefSet;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.function.Function;

public @RefAware
class CudaDevice extends CudaSystem {
  protected static final Logger logger = LoggerFactory.getLogger(CudnnHandle.class);
  @Nullable
  protected final String deviceName;
  protected final int deviceId;
  final Object allocationLock = new Object();
  private final Object memoryManagementLock = new Object();
  private volatile cudaDeviceProp deviceProperties;

  protected CudaDevice(final int deviceId) {
    super();
    this.deviceId = deviceId;
    assert 0 <= this.deviceId;
    if (0 <= this.deviceId) {
      initThread();
      deviceName = getDeviceName(deviceId);
    } else {
      deviceName = null;
    }
  }

  public int getDeviceId() {
    return deviceId;
  }

  public static void setDevice(final int cudaDeviceId) {
    if (cudaDeviceId < 0)
      throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
    if (!isThreadDeviceId(cudaDeviceId)) {
      long startTime = System.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((System.nanoTime() - startTime) / 1e9);
      log("cudaSetDevice", result, new Object[]{cudaDeviceId});
      handle(result);
      currentDeviceId.set(cudaDeviceId);
    }
  }

  public static void cudaFree(int deviceId, final CudaPointer devPtr) {
    long startTime = System.nanoTime();
    if (null == devPtr)
      return;
    Function<CudnnHandle, Integer> fn = dev -> {
      final int result = JCuda.cudaFree(devPtr);
      log("cudaFree", result, new Object[]{devPtr});
      cudaFree_execution.accept((System.nanoTime() - startTime) / 1e9);
      handle(result);
      return result;
    };
    if (deviceId < 0) {
      fn.apply(null);
    } else {
      withDevice(deviceId, fn);
    }
  }

  public static String getDeviceName(final int device) {
    return new String(CudaDevice.getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }

  public static cudaDeviceProp getDeviceProperties(final int device) {
    return propertyCache.computeIfAbsent(device, deviceId -> {
      long startTime = System.nanoTime();
      @Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((System.nanoTime() - startTime) / 1e9);
      log("cudaGetDeviceProperties", result, new Object[]{deviceProp, device});
      return deviceProp;
    });
  }

  @Nonnull
  public DeviceMetrics ensureCapacity(final long size) {
    if (size <= 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaSettings.INSTANCE().getMaxAllocSize()) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    final DeviceMetrics metrics;
    synchronized (memoryManagementLock) {
      metrics = CudaMemory.getGpuStats(deviceId);
      RefCollection<DeviceMetrics> temp_75_0001 = CudaMemory.METRICS
          .values();
      double resultingTotalMemory = temp_75_0001.stream().mapToLong(m -> m.usedMemory.get()).sum() + size;
      if (null != temp_75_0001)
        temp_75_0001.freeRef();
      if (resultingTotalMemory > CudaSettings.INSTANCE().getMaxTotalMemory()) {
        CudaMemory.logger.info(String.format("Clearing weak global memory while allocating %e bytes (%e > %e)",
            (double) size, resultingTotalMemory, CudaSettings.INSTANCE().getMaxTotalMemory()));
        CudaMemory.clearWeakMemory(deviceId);
      }
      RefCollection<DeviceMetrics> temp_75_0002 = CudaMemory.METRICS
          .values();
      resultingTotalMemory = temp_75_0002.stream().mapToLong(x1 -> x1.usedMemory.get()).sum() + size;
      if (null != temp_75_0002)
        temp_75_0002.freeRef();
      if (resultingTotalMemory > CudaSettings.INSTANCE().getMaxTotalMemory()) {
        CudaMemory.logger.info(String.format("Clearing all global memory while allocating %e bytes (%e > %e)",
            (double) size, resultingTotalMemory, CudaSettings.INSTANCE().getMaxTotalMemory()));
        CudaMemory.clearMemory(deviceId);
      }
      double resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > CudaSettings.INSTANCE().getMaxDeviceMemory()) {
        CudaMemory.logger.info(String.format("Clearing weak memory for device %s while allocating %e bytes (%e > %e)",
            this, (double) size, resultingDeviceMemory, CudaSettings.INSTANCE().getMaxDeviceMemory()));
        RefSet<Integer> temp_75_0003 = CudaMemory.METRICS.keySet();
        temp_75_0003.stream().mapToInt(x -> x).distinct().forEach(CudaMemory::clearWeakMemory);
        if (null != temp_75_0003)
          temp_75_0003.freeRef();
      }
      resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > CudaSettings.INSTANCE().getMaxDeviceMemory()) {
        CudaMemory.logger.info(String.format("Clearing all memory for device %s while allocating %e bytes (%s > %e)",
            this, (double) size, resultingDeviceMemory, CudaSettings.INSTANCE().getMaxDeviceMemory()));
        RefSet<Integer> temp_75_0004 = CudaMemory.METRICS.keySet();
        temp_75_0004.stream().mapToInt(x -> x).distinct().forEach(CudaMemory::clearMemory);
        if (null != temp_75_0004)
          temp_75_0004.freeRef();
      }
    }
    return metrics;
  }

  @Nonnull
  public CudaMemory allocate(final long size, @Nonnull MemoryType type, boolean dirty) {
    assert isThreadDeviceId(getDeviceId());
    @Nonnull
    CudaMemory obtain = new CudaMemory(this, size, type);
    if (!dirty)
      obtain.clear();
    return obtain;
  }

  public CudaTensorDescriptor newTensorDescriptor(final Precision dataType, final int batchCount, final int channels,
                                                  final int height, final int width) {
    return newTensorDescriptor(dataType, batchCount, channels, height, width, channels * height * width, height * width,
        width, 1);
  }

  public CudaTensorDescriptor newTensorDescriptor(final Precision dataType, final int batchCount, final int channels,
                                                  final int height, final int width, final int nStride, final int cStride, final int hStride, final int wStride) {
    assert batchCount > 0;
    assert channels > 0;
    assert height > 0;
    assert width > 0;
    assert nStride > 0;
    assert cStride > 0;
    assert hStride != 0;
    assert wStride != 0;
    long startTime = System.nanoTime();
    @Nonnull final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    log("cudnnCreateTensorDescriptor", result, new Object[]{desc});
    handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType.code, batchCount, channels, height, width, nStride,
        cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetTensor4dDescriptorEx", result,
        new Object[]{desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride});
    handle(result);
    return new CudaTensorDescriptor(desc, getDeviceId(), dataType, batchCount, channels, height, width, nStride,
        cStride, hStride, wStride);
  }

  public CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(final int opType, final Precision dataType) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    log("cudnnCreateOpTensorDescriptor", result, new Object[]{opDesc});
    handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType.code,
        cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetOpTensorDescriptor", result,
        new Object[]{opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN});
    handle(result);
    return new CudaResource<>(opDesc, CudaSystem::cudnnDestroyOpTensorDescriptor, getDeviceId());
  }

  public CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final Precision dataType, final int tensorLayout,
                                                                 final int outputChannels, final int inputChannels, final int height, final int width) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, new Object[]{filterDesc});
    handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType.code, tensorLayout, outputChannels, inputChannels,
        height, width);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetFilter4dDescriptor", result,
        new Object[]{filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width});
    handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CudaSystem::cudnnDestroyFilterDescriptor,
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
      }
    };
  }

  public CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int mode, final Precision dataType,
                                                                              final int paddingY, final int paddingX, final int strideHeight, final int strideWidth, int dilationY,
                                                                              int dilationX) {
    long startTime = System.nanoTime();
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
    newConvolutions2dDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetConvolution2dDescriptor", result,
        new Object[]{convDesc, paddingY, paddingX, strideHeight, strideWidth, dilationY, dilationX, mode, dataType});
    handle(result);
    return new CudaResource<>(convDesc, CudaSystem::cudnnDestroyConvolutionDescriptor, getDeviceId());
  }

  public CudaResource<cudnnActivationDescriptor> newActivationDescriptor(final int mode, final int reluNan,
                                                                         final double reluCeil) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    log("cudnnCreateActivationDescriptor", result, new Object[]{desc});
    handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetActivationDescriptor", result, new Object[]{desc, mode, reluNan, reluCeil});
    handle(result);
    return new CudaResource<>(desc, CudaSystem::cudnnDestroyActivationDescriptor, getDeviceId());
  }

  public CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(final int mode, final int poolDims,
                                                                      final int[] windowSize, final int[] padding, final int[] stride) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    log("cudnnCreatePoolingDescriptor", result, new Object[]{poolingDesc});
    handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc, mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        poolDims, windowSize, padding, stride);
    log("cudnnSetPoolingNdDescriptor", result, new Object[]{poolingDesc, mode,
        cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize, padding, stride});
    handle(result);
    createPoolingDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, CudaSystem::cudnnDestroyPoolingDescriptor, getDeviceId());
  }

  public CudaResource<cudnnLRNDescriptor> createLRNDescriptor(int lrnN, double lrnAlpha, double lrnBeta, double lrnK) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnLRNDescriptor poolingDesc = new cudnnLRNDescriptor();
    int result = JCudnn.cudnnCreateLRNDescriptor(poolingDesc);
    log("cudnnCreateLRNDescriptor", result, new Object[]{poolingDesc});
    handle(result);
    result = JCudnn.cudnnSetLRNDescriptor(poolingDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
    log("cudnnSetLRNDescriptor", result, new Object[]{poolingDesc, lrnN, lrnAlpha, lrnBeta, lrnK});
    handle(result);
    createLRNDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, JCudnn::cudnnDestroyLRNDescriptor, getDeviceId());
  }

  public void initThread() {
    setDevice(getDeviceId());
  }

  @Nonnull
  CudaPointer acquire(long size, @Nonnull MemoryType type, int retries) {
    if (size <= 0)
      throw new IllegalArgumentException();
    assert isThreadDeviceId(getDeviceId());
    if (retries < 0)
      throw new IllegalArgumentException();
    @Nonnull
    CudaPointer pointer = _acquire(size, type, retries);
    if (null != pointer)
      return pointer;
    if (retries < 0)
      throw new IllegalStateException();
    return this.acquire(size, type, retries - 1);
  }

  private CudaPointer _acquire(long size, @Nonnull MemoryType type, int retries) {
    synchronized (allocationLock) {
      final DeviceMetrics metrics = ensureCapacity(size);
      @Nonnull
      CudaPointer pointer = null;
      try {
        pointer = type.allocCached(size, this);
        final long finalMemory = metrics.activeMemory.addAndGet(size);
        metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      } catch (@Nonnull final ThreadDeath e) {
        throw e;
      } catch (@Nonnull final Throwable e) {
        if (retries <= 0)
          throw new RuntimeException(
              String.format(String.format("Error allocating %e bytes; %s currently allocated to device %s",
                  (double) size, metrics.usedMemory, this)),
              e);
        final long startMemory = metrics.usedMemory.get();
        @Nonnull
        TimedResult<Double> timedResult = TimedResult.time(() -> CudaMemory.clearMemory(getDeviceId()));
        final long freedMemory = startMemory - metrics.usedMemory.get();
        CudaMemory.logger.warn(String.format(
            "Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)", size,
            freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
      }
      return pointer;
    }
  }

  public static @RefAware
  class CudaTensorDescriptor extends CudaResource<cudnnTensorDescriptor> {

    public final int wStride;
    public final int hStride;
    public final int cStride;
    public final int nStride;
    public final int width;
    public final int height;
    public final int channels;
    public final int batchCount;
    public final Precision dataType;

    protected CudaTensorDescriptor(final cudnnTensorDescriptor obj, final int deviceId, final Precision dataType,
                                   final int batchCount, final int channels, final int height, final int width, final int nStride,
                                   final int cStride, final int hStride, final int wStride) {
      super(obj, CudaSystem::cudnnDestroyTensorDescriptor, deviceId);
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

    public static @SuppressWarnings("unused")
    CudaTensorDescriptor[] addRefs(CudaTensorDescriptor[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(CudaTensorDescriptor::addRef)
          .toArray((x) -> new CudaTensorDescriptor[x]);
    }

    public CudaTensorDescriptor copy(CudaDevice device) {
      return device.newTensorDescriptor(dataType, batchCount, channels, height, width, nStride, cStride, hStride,
          wStride);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    CudaTensorDescriptor addRef() {
      return (CudaTensorDescriptor) super.addRef();
    }
  }
}
