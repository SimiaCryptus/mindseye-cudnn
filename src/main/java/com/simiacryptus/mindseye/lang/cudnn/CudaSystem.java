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

import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.lang.ResourcePool;
import com.simiacryptus.lang.StaticResourcePool;
import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.DoubleStatistics;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaStream_t;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * The type Cuda system.
 */
public class CudaSystem extends ReferenceCountingBase {

  /**
   * The constant apiLog.
   */
  public static final RefHashSet<RefConsumer<String>> apiLog = new RefHashSet<>();
  /**
   * The constant gpuGeneration.
   */
  @Nonnull
  public static final AtomicInteger gpuGeneration = new AtomicInteger(0);
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaSystem.class);
  /**
   * The constant propertyCache.
   */
  protected static final RefMap<Integer, cudaDeviceProp> propertyCache = new RefConcurrentHashMap<>();
  /**
   * The constant currentDeviceId.
   */
  protected static final ThreadLocal<Integer> currentDeviceId = new ThreadLocal<Integer>();
  /**
   * The constant logThread.
   */
  protected static final ExecutorService logThread = Executors
      .newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The constant start.
   */
  protected static final long start = RefSystem.nanoTime();
  /**
   * The constant createPoolingDescriptor_execution.
   */
  protected static final DoubleStatistics createPoolingDescriptor_execution = new DoubleStatistics();
  /**
   * The constant createLRNDescriptor_execution.
   */
  protected static final DoubleStatistics createLRNDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceReset_execution.
   */
  protected static final DoubleStatistics cudaDeviceReset_execution = new DoubleStatistics();
  /**
   * The constant cudaFree_execution.
   */
  protected static final DoubleStatistics cudaFree_execution = new DoubleStatistics();
  /**
   * The constant cudaMalloc_execution.
   */
  protected static final DoubleStatistics cudaMalloc_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceSynchronize_execution.
   */
  protected static final DoubleStatistics cudaDeviceSynchronize_execution = new DoubleStatistics();
  /**
   * The constant cudaSetDeviceFlags_execution.
   */
  protected static final DoubleStatistics cudaSetDeviceFlags_execution = new DoubleStatistics();
  /**
   * The constant cudaMallocManaged_execution.
   */
  protected static final DoubleStatistics cudaMallocManaged_execution = new DoubleStatistics();
  /**
   * The constant cudaHostAlloc_execution.
   */
  protected static final DoubleStatistics cudaHostAlloc_execution = new DoubleStatistics();
  /**
   * The constant cudaFreeHost_execution.
   */
  protected static final DoubleStatistics cudaFreeHost_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceGetLimit_execution.
   */
  protected static final DoubleStatistics cudaDeviceGetLimit_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceSetLimit_execution.
   */
  protected static final DoubleStatistics cudaDeviceSetLimit_execution = new DoubleStatistics();
  /**
   * The constant cudaMemcpyAsync_execution.
   */
  protected static final DoubleStatistics cudaMemcpyAsync_execution = new DoubleStatistics();
  /**
   * The constant cudaMemcpy_execution.
   */
  protected static final DoubleStatistics cudaMemcpy_execution = new DoubleStatistics();
  /**
   * The constant cudaMemset_execution.
   */
  protected static final DoubleStatistics cudaMemset_execution = new DoubleStatistics();
  /**
   * The constant cudnnSoftmaxForward_execution.
   */
  protected static final DoubleStatistics cudnnSoftmaxForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnSoftmaxBackward_execution.
   */
  protected static final DoubleStatistics cudnnSoftmaxBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnCreateReduceTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnCreateReduceTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnSetReduceTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnSetReduceTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnActivationBackward_execution.
   */
  protected static final DoubleStatistics cudnnActivationBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnActivationForward_execution.
   */
  protected static final DoubleStatistics cudnnActivationForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnAddTensor_execution.
   */
  protected static final DoubleStatistics cudnnAddTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardBias_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardBias_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardData_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardData_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardFilter_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardFilter_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionForward_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBiasActivationForward_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBiasActivationForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyActivationDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyActivationDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyConvolutionDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyConvolutionDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyFilterDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyFilterDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyOpTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyOpTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyPoolingDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyPoolingDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnGetPoolingNdForwardOutputDim_execution.
   */
  protected static final DoubleStatistics cudnnGetPoolingNdForwardOutputDim_execution = new DoubleStatistics();
  /**
   * The constant cudnnOpTensor_execution.
   */
  protected static final DoubleStatistics cudnnOpTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnReduceTensor_execution.
   */
  protected static final DoubleStatistics cudnnReduceTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnPoolingBackward_execution.
   */
  protected static final DoubleStatistics cudnnPoolingBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnPoolingForward_execution.
   */
  protected static final DoubleStatistics cudnnPoolingForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnSetLRNDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnSetLRNDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnCreateLRNDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnCreateLRNDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyLRNDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyLRNDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnLRNCrossChannelForward_execution.
   */
  protected static final DoubleStatistics cudnnLRNCrossChannelForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnLRNCrossChannelBackward_execution.
   */
  protected static final DoubleStatistics cudnnLRNCrossChannelBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnTransformTensor_execution.
   */
  protected static final DoubleStatistics cudnnTransformTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnSetTensor_execution.
   */
  protected static final DoubleStatistics cudnnSetTensor_execution = new DoubleStatistics();
  /**
   * The constant deviceCount_execution.
   */
  protected static final DoubleStatistics deviceCount_execution = new DoubleStatistics();
  /**
   * The constant setDevice_execution.
   */
  protected static final DoubleStatistics setDevice_execution = new DoubleStatistics();
  /**
   * The constant getDeviceProperties_execution.
   */
  protected static final DoubleStatistics getDeviceProperties_execution = new DoubleStatistics();
  /**
   * The constant getOutputDims_execution.
   */
  protected static final DoubleStatistics getOutputDims_execution = new DoubleStatistics();
  /**
   * The constant newActivationDescriptor_execution.
   */
  protected static final DoubleStatistics newActivationDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newConvolutionNdDescriptor_execution.
   */
  protected static final DoubleStatistics newConvolutionNdDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newConvolutions2dDescriptor_execution.
   */
  protected static final DoubleStatistics newConvolutions2dDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newFilterDescriptor_execution.
   */
  protected static final DoubleStatistics newFilterDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newOpDescriptor_execution.
   */
  protected static final DoubleStatistics newOpDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newTensorDescriptor_execution.
   */
  protected static final DoubleStatistics newTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant allocateBackwardDataWorkspace_execution.
   */
  protected static final DoubleStatistics allocateBackwardDataWorkspace_execution = new DoubleStatistics();
  /**
   * The constant allocateBackwardFilterWorkspace_execution.
   */
  protected static final DoubleStatistics allocateBackwardFilterWorkspace_execution = new DoubleStatistics();
  /**
   * The constant allocateForwardWorkspace_execution.
   */
  protected static final DoubleStatistics allocateForwardWorkspace_execution = new DoubleStatistics();
  /**
   * The constant getBackwardDataAlgorithm_execution.
   */
  protected static final DoubleStatistics getBackwardDataAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant getBackwardFilterAlgorithm_execution.
   */
  protected static final DoubleStatistics getBackwardFilterAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamCreate_execution.
   */
  protected static final DoubleStatistics cudaStreamCreate_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamDestroy_execution.
   */
  protected static final DoubleStatistics cudaStreamDestroy_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamSynchronize_execution.
   */
  protected static final DoubleStatistics cudaStreamSynchronize_execution = new DoubleStatistics();
  /**
   * The constant getForwardAlgorithm_execution.
   */
  protected static final DoubleStatistics getForwardAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant handlePools.
   */
  protected static final RefConcurrentHashMap<Integer, ResourcePool<CudnnHandle>> handlePools = new RefConcurrentHashMap<>();
  private static final Map<Integer, Long> syncTimes = new HashMap<>();
  private static final long COPY_BLOCK_SIZE = Long.MAX_VALUE;
  private static volatile Integer cachedDeviceCount = init();
  private static volatile StaticResourcePool<CudnnHandle> pool;
  /**
   * The Execution thread.
   */
  protected final ExecutorService executionThread = CoreSettings.INSTANCE().singleThreaded
      ? MoreExecutors.newDirectExecutorService()
      : Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setNameFormat(toString()).build());

  /**
   * Instantiates a new Cuda system.
   */
  protected CudaSystem() {
  }

  /**
   * Gets cached device count.
   *
   * @return the cached device count
   */
  public static int getCachedDeviceCount() {
    if (null == cachedDeviceCount) {
      synchronized (CudaSystem.class) {
        if (null == cachedDeviceCount) {
          cachedDeviceCount = init();
        }
      }
    }
    return cachedDeviceCount;
  }

  private static int getDeviceCount() {
    final int deviceCount;
    if (CudaSettings.INSTANCE().forceSingleGpu) {
      CudaDevice.logger.warn("Forcing Single-GPU Mode");
      deviceCount = 1;
    } else {
      deviceCount = CudaSystem.deviceCount();
    }
    CudaDevice.logger.info(RefString.format("Found %s devices", deviceCount));
    return deviceCount;
  }

  /**
   * Gets execution statistics.
   *
   * @return the execution statistics
   */
  @Nonnull
  @RefIgnore
  public static HashMap<CharSequence, Map<CharSequence, CharSequence>> getExecutionStatistics() {
    @Nonnull
    HashMap<CharSequence, Map<CharSequence, CharSequence>> map = new HashMap<>();
    map.put("createPoolingDescriptor", toMap(createPoolingDescriptor_execution));
    map.put("cudaDeviceReset", toMap(cudaDeviceReset_execution));
    map.put("cudaFree", toMap(cudaFree_execution));
    map.put("cudaMalloc", toMap(cudaMalloc_execution));
    map.put("cudaMallocManaged", toMap(cudaMallocManaged_execution));
    map.put("cudaHostAlloc", toMap(cudaHostAlloc_execution));
    map.put("cudaFreeHost", toMap(cudaFreeHost_execution));
    map.put("cudaDeviceGetLimit", toMap(cudaDeviceGetLimit_execution));
    map.put("cudaDeviceSetLimit", toMap(cudaDeviceSetLimit_execution));
    map.put("cudaMemcpy", toMap(cudaMemcpy_execution));
    map.put("cudaMemset", toMap(cudaMemset_execution));
    map.put("cudnnActivationBackward", toMap(cudnnActivationBackward_execution));
    map.put("cudnnActivationForward", toMap(cudnnActivationForward_execution));
    map.put("cudnnAddTensor", toMap(cudnnAddTensor_execution));
    map.put("cudnnConvolutionBackwardBias", toMap(cudnnConvolutionBackwardBias_execution));
    map.put("cudnnConvolutionBackwardData", toMap(cudnnConvolutionBackwardData_execution));
    map.put("cudnnConvolutionBackwardFilter", toMap(cudnnConvolutionBackwardFilter_execution));
    map.put("cudnnConvolutionForward", toMap(cudnnConvolutionForward_execution));
    map.put("cudnnDestroyActivationDescriptor", toMap(cudnnDestroyActivationDescriptor_execution));
    map.put("cudnnDestroyConvolutionDescriptor", toMap(cudnnDestroyConvolutionDescriptor_execution));
    map.put("cudnnDestroyFilterDescriptor", toMap(cudnnDestroyFilterDescriptor_execution));
    map.put("cudnnDestroyOpTensorDescriptor", toMap(cudnnDestroyOpTensorDescriptor_execution));
    map.put("cudnnDestroyPoolingDescriptor", toMap(cudnnDestroyPoolingDescriptor_execution));
    map.put("cudnnDestroyTensorDescriptor", toMap(cudnnDestroyTensorDescriptor_execution));
    map.put("cudnnGetPoolingNdForwardOutputDim", toMap(cudnnGetPoolingNdForwardOutputDim_execution));
    map.put("cudnnOpTensor", toMap(cudnnOpTensor_execution));
    map.put("cudnnPoolingBackward", toMap(cudnnPoolingBackward_execution));
    map.put("cudnnPoolingForward", toMap(cudnnPoolingForward_execution));
    map.put("cudnnTransformTensor", toMap(cudnnTransformTensor_execution));
    map.put("cachedDeviceCount", toMap(deviceCount_execution));
    map.put("setDevice", toMap(setDevice_execution));
    map.put("getDeviceProperties", toMap(getDeviceProperties_execution));
    map.put("getOutputDims", toMap(getOutputDims_execution));
    map.put("newActivationDescriptor", toMap(newActivationDescriptor_execution));
    map.put("newConvolutionNdDescriptor", toMap(newConvolutionNdDescriptor_execution));
    map.put("newConvolutions2dDescriptor", toMap(newConvolutions2dDescriptor_execution));
    map.put("newFilterDescriptor", toMap(newFilterDescriptor_execution));
    map.put("newOpDescriptor", toMap(newOpDescriptor_execution));
    map.put("newTensorDescriptor", toMap(newTensorDescriptor_execution));
    map.put("allocateBackwardDataWorkspace", toMap(allocateBackwardDataWorkspace_execution));
    map.put("allocateBackwardFilterWorkspace", toMap(allocateBackwardFilterWorkspace_execution));
    map.put("allocateForwardWorkspace", toMap(allocateForwardWorkspace_execution));
    map.put("getBackwardDataAlgorithm", toMap(getBackwardDataAlgorithm_execution));
    map.put("getBackwardFilterAlgorithm", toMap(getBackwardFilterAlgorithm_execution));
    map.put("getForwardAlgorithm", toMap(getForwardAlgorithm_execution));
    map.put("cudaDeviceSynchronize", toMap(cudaDeviceSynchronize_execution));
    map.put("cudaStreamCreate", toMap(cudaStreamCreate_execution));
    map.put("cudaStreamDestroy", toMap(cudaStreamDestroy_execution));
    map.put("cudaStreamSynchronize", toMap(cudaStreamSynchronize_execution));
    map.put("cudaMemcpyAsync", toMap(cudaMemcpyAsync_execution));
    map.put("cudaSetDeviceFlags", toMap(cudaSetDeviceFlags_execution));

    List<CharSequence> list = map.entrySet().stream().filter(x -> {
      return x.getValue().isEmpty();
    }).map(x -> {
      return x.getKey();
    }).collect(Collectors.toList());
    list.stream().forEach(value -> map.remove(value));
    return map;
  }

  /**
   * Gets thread device id.
   *
   * @return the thread device id
   */
  public static Integer getThreadDeviceId() {
    return CudaSystem.currentDeviceId.get();
  }

  /**
   * Is enabled boolean.
   *
   * @return the boolean
   */
  public static boolean isEnabled() {
    return 0 < getCachedDeviceCount();
  }

  /**
   * Print header.
   *
   * @param out the out
   */
  public static void printHeader(@Nonnull PrintStream out) {
    @Nonnull
    int[] runtimeVersion = {0};
    @Nonnull
    int[] driverVersion = {0};
    JCuda.cudaRuntimeGetVersion(runtimeVersion);
    JCuda.cudaDriverGetVersion(driverVersion);
    @Nonnull
    CharSequence jCudaVersion = JCuda.getJCudaVersion();
    out.printf("Time: %s; Driver %s; Runtime %s; Lib %s%n", new Date(), driverVersion[0], runtimeVersion[0],
        jCudaVersion);
    @Nonnull
    long[] free = {0};
    @Nonnull
    long[] total = {0};
    JCuda.cudaMemGetInfo(free, total);
    out.printf("Cuda Memory: %.1f free, %.1f total%n", free[0] * 1.0 / (1024 * 1024), total[0] * 1.0 / (1024 * 1024));
    @Nonnull final int[] deviceCount = new int[1];
    JCuda.cudaGetDeviceCount(deviceCount);
    RefIntStream.range(0, deviceCount[0]).forEach(device -> {
      @Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      JCuda.cudaGetDeviceProperties(deviceProp, device);
      out.printf("Device %d = %s%n", device, deviceProp, free[0], total[0]);
    });
    out.printf("CuDNN version: %d / %d%n", JCudnn.cudnnGetVersion(), JCudnn.cudnnGetCudartVersion());
    RefSystem.getProperties().forEach((k, v) -> {
      boolean display = false;
      if (k.toString().endsWith(".version"))
        display = true;
      if (k.toString().startsWith("os."))
        display = true;
      if (k.toString().contains("arch"))
        display = true;
      if (display)
        out.printf("%s = %s%n", k, v);
    });
  }

  /**
   * Cuda device reset int.
   *
   * @return the int
   */
  public static int cudaDeviceReset() {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaDeviceReset();
    log("cudaDeviceReset", result, new Object[]{});
    cudaDeviceReset_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }

  /**
   * Cuda malloc int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @return the int
   */
  public static int cudaMalloc(final CudaPointer devPtr, final long size) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaMalloc(devPtr, size);
    log("cudaMalloc", result, new Object[]{devPtr, size});
    cudaMalloc_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }

  /**
   * Cuda malloc managed int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @param flags  the flags
   * @return the int
   */
  public static int cudaMallocManaged(final CudaPointer devPtr, final long size, int flags) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaMallocManaged(devPtr, size, flags);
    log("cudaMallocManaged", result, new Object[]{devPtr, size, flags});
    cudaMallocManaged_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }

  /**
   * Cuda set device flags int.
   *
   * @param flags the flags
   * @return the int
   */
  public static int cudaSetDeviceFlags(int flags) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaSetDeviceFlags(flags);
    log("cudaSetDeviceFlags", result, new Object[]{flags});
    cudaDeviceSynchronize_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }

  /**
   * Cuda host alloc int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @param flags  the flags
   * @return the int
   */
  public static int cudaHostAlloc(final CudaPointer devPtr, final long size, int flags) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaHostAlloc(devPtr, size, flags);
    cudaHostAlloc_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaHostAlloc", result, new Object[]{devPtr, size, flags});
    handle(result);
    return result;
  }

  /**
   * Cuda free host.
   *
   * @param devPtr the dev ptr
   */
  public static void cudaFreeHost(final CudaPointer devPtr) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaFreeHost(devPtr);
    cudaFreeHost_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaFreeHost", result, new Object[]{devPtr});
    handle(result);
  }

  /**
   * Cuda device get limit long.
   *
   * @param limit the limit
   * @return the long
   */
  public static long cudaDeviceGetLimit(final int limit) {
    long startTime = RefSystem.nanoTime();
    @Nonnull
    long[] pValue = new long[1];
    final int result = JCuda.cudaDeviceGetLimit(pValue, limit);
    cudaDeviceGetLimit_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaDeviceGetLimit(", result, new Object[]{pValue, limit});
    return pValue[0];
  }

  /**
   * Cuda device set limit.
   *
   * @param limit the limit
   * @param value the value
   */
  public static void cudaDeviceSetLimit(final int limit, long value) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaDeviceSetLimit(limit, value);
    cudaDeviceSetLimit_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaDeviceSetLimit(", result, new Object[]{limit, value});
    handle(result);
  }

  /**
   * Cuda memcpy.
   *
   * @param dst                 the dst
   * @param src                 the src
   * @param count               the count
   * @param cudaMemcpyKind_kind the cuda memcpy kind kind
   */
  public static void cudaMemcpy(@Nonnull final CudaPointer dst, @Nonnull final CudaPointer src, final long count,
                                final int cudaMemcpyKind_kind) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    cudaMemcpy_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaMemcpy", result, new Object[]{dst, src, count, cudaMemcpyKind_kind});
    handle(result);
  }

  /**
   * Cuda memcpy async.
   *
   * @param dst                 the dst
   * @param src                 the src
   * @param count               the count
   * @param cudaMemcpyKind_kind the cuda memcpy kind kind
   * @param stream              the stream
   */
  public static void cudaMemcpyAsync(final CudaPointer dst, final CudaPointer src, final long count,
                                     final int cudaMemcpyKind_kind, cudaStream_t stream) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaMemcpyAsync(dst, src, count, cudaMemcpyKind_kind, stream);
    cudaMemcpyAsync_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaMemcpyAsync", result, new Object[]{dst, src, count, cudaMemcpyKind_kind, stream});
    handle(result);
  }

  /**
   * Cuda stream create cuda resource.
   *
   * @return the cuda resource
   */
  @Nonnull
  public static CudaResource<cudaStream_t> cudaStreamCreate() {
    long startTime = RefSystem.nanoTime();
    @Nonnull
    cudaStream_t stream = new cudaStream_t();
    int result = JCuda.cudaStreamCreate(stream);
    cudaStreamCreate_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaStreamCreate", result, new Object[]{stream});
    handle(result);
    return new CudaStream(stream);
  }

  /**
   * Cuda stream destroy int.
   *
   * @param stream the stream
   * @return the int
   */
  public static int cudaStreamDestroy(cudaStream_t stream) {
    long startTime = RefSystem.nanoTime();
    int result = JCuda.cudaStreamDestroy(stream);
    cudaStreamDestroy_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaStreamDestroy", result, new Object[]{stream});
    handle(result);
    return result;
  }

  /**
   * Cuda stream synchronize.
   *
   * @param stream the stream
   */
  public static void cudaStreamSynchronize(cudaStream_t stream) {
    long startTime = RefSystem.nanoTime();
    int result = JCuda.cudaStreamSynchronize(stream);
    cudaStreamSynchronize_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaStreamSynchronize", result, new Object[]{stream});
    handle(result);
  }

  /**
   * Cuda memset.
   *
   * @param mem   the mem
   * @param c     the c
   * @param count the count
   */
  public static void cudaMemset(@Nonnull final CudaPointer mem, final int c, final long count) {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaMemset(mem, c, count);
    //cudaDeviceSynchronize();
    cudaMemset_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudaMemset", result, new Object[]{mem, c, count});
    handle(result);
  }

  /**
   * Cudnn destroy activation descriptor int.
   *
   * @param activationDesc the activation desc
   * @return the int
   */
  public static int cudnnDestroyActivationDescriptor(final cudnnActivationDescriptor activationDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyActivationDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyActivationDescriptor", result, new Object[]{activationDesc});
    return result;
  }

  /**
   * Cudnn destroy convolution descriptor int.
   *
   * @param convDesc the conv desc
   * @return the int
   */
  public static int cudnnDestroyConvolutionDescriptor(final cudnnConvolutionDescriptor convDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyConvolutionDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyConvolutionDescriptor", result, new Object[]{convDesc});
    return result;
  }

  /**
   * Cudnn destroy filter descriptor int.
   *
   * @param filterDesc the filter desc
   * @return the int
   */
  public static int cudnnDestroyFilterDescriptor(final cudnnFilterDescriptor filterDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyFilterDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyFilterDescriptor", result, new Object[]{filterDesc});
    return result;
  }

  /**
   * Cudnn destroy op tensor descriptor int.
   *
   * @param opTensorDesc the op tensor desc
   * @return the int
   */
  public static int cudnnDestroyOpTensorDescriptor(final cudnnOpTensorDescriptor opTensorDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyOpTensorDescriptor(opTensorDesc);
    cudnnDestroyOpTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyOpTensorDescriptor", result, new Object[]{opTensorDesc});
    return result;
  }

  /**
   * Cudnn destroy pooling descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @return the int
   */
  public static int cudnnDestroyPoolingDescriptor(final cudnnPoolingDescriptor poolingDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
    cudnnDestroyPoolingDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyPoolingDescriptor", result, new Object[]{poolingDesc});
    return result;
  }

  /**
   * Cudnn destroy tensor descriptor int.
   *
   * @param tensorDesc the tensor desc
   * @return the int
   */
  public static int cudnnDestroyTensorDescriptor(final cudnnTensorDescriptor tensorDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroyTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyTensorDescriptor", result, new Object[]{tensorDesc});
    return result;
  }

  /**
   * Cudnn get pooling nd forward output dim int.
   *
   * @param poolingDesc      the pooling desc
   * @param inputTensorDesc  the input tensor desc
   * @param nbDims           the nb dims
   * @param outputTensorDimA the output tensor dim a
   * @return the int
   */
  public static int cudnnGetPoolingNdForwardOutputDim(final cudnnPoolingDescriptor poolingDesc,
                                                      final cudnnTensorDescriptor inputTensorDesc, final int nbDims, final int[] outputTensorDimA) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    cudnnGetPoolingNdForwardOutputDim_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetPoolingNdForwardOutputDim", result,
        new Object[]{poolingDesc, inputTensorDesc, nbDims, outputTensorDimA});
    return result;
  }

  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int[] deviceCount = new int[1];
    final int returnCode = JCuda.cudaGetDeviceCount(deviceCount);
    log("cudaGetDeviceCount", returnCode, new Object[]{deviceCount});
    deviceCount_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    CudaSystem.handle(returnCode);
    return deviceCount[0];
  }

  /**
   * Handle.
   *
   * @param returnCode the return code
   */
  public static void handle(final int returnCode) {
    if (returnCode != cudnnStatus.CUDNN_STATUS_SUCCESS) {
      CudaError cudaError = new CudaError("returnCode = " + cudnnStatus.stringFor(returnCode));
      logger.warn("Cuda Error", cudaError);
      throw cudaError;
    }
  }

  /**
   * Get output dims int [ ].
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @return the int [ ]
   */
  @Nonnull
  public static int[] getOutputDims(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc,
                                    final cudnnConvolutionDescriptor convDesc) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int[] tensorOuputDims = new int[4];
    final int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc,
        tensorOuputDims.length, tensorOuputDims);
    getOutputDims_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionNdForwardOutputDim", result,
        new Object[]{convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims});
    CudaSystem.handle(result);
    return tensorOuputDims;
  }

  /**
   * Add log.
   *
   * @param log the log
   */
  public static void addLog(@Nonnull PrintStream log) {
    printHeader(log);
    apiLog.add(s -> log.println(s));
  }

  /**
   * Add log.
   *
   * @param log the log
   */
  public static void addLog(@Nonnull RefConsumer<String> log) {
    apiLog.add(log);
  }

  /**
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  public static void log(final CharSequence method, @RefAware final Object result, @Nullable @RefAware final Object[] args) {
    if (CudaSystem.apiLog.isEmpty()) {
      RefUtil.freeRef(result);
      RefUtil.freeRef(args);
      return;
    }
    CharSequence callstack = !CudaSettings.INSTANCE().logStack ? ""
        : Util.toString(RefArrays.stream(Thread.currentThread().getStackTrace())
        .filter(x -> true && x.getClassName().startsWith("com.simiacryptus.mindseye.")
            //&& !x.getClassName().startsWith("com.simiacryptus.mindseye.lang.")
            //&& !x.getClassName().startsWith("com.simiacryptus.mindseye.test.")
        )
        //.limit(10)
        .toArray(i -> new StackTraceElement[i]), ", ");
    @Nonnull final CharSequence paramString;
    if (null == args) paramString = "";
    else paramString = RefArrays.stream(args).map(obj -> renderToLog(obj)).reduce((a, b) -> a + ", " + b).orElse("");
    final String message = RefString.format("%.6f @ %s(%d): %s(%s) = %s via [%s]",
        (RefSystem.nanoTime() - CudaSystem.start) / 1e9, Thread.currentThread().getName(),
        getThreadDeviceId(), method, paramString, result, callstack);
    try {
      CudaSystem.apiLog.forEach(apiLog -> CudaSystem.logThread.submit(() -> apiLog.accept(message)));
    } catch (ConcurrentModificationException e) {
    }
  }

  /**
   * Is thread device id boolean.
   *
   * @param deviceId the device id
   * @return the boolean
   */
  public static boolean isThreadDeviceId(int deviceId) {
    Integer integer = getThreadDeviceId();
    return integer != null && deviceId == integer;
  }

  /**
   * With device.
   *
   * @param deviceId the device id
   * @param fn       the fn
   */
  public static void withDevice(int deviceId, @Nonnull @RefAware final RefConsumer<CudnnHandle> fn) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null && threadlocal.getDeviceId() == deviceId) {
        assert CudaSystem.isThreadDeviceId(threadlocal.getDeviceId());
        try {
          fn.accept(threadlocal.addRef());
        } finally {
          RefUtil.freeRef(fn);
        }
      } else {
        ResourcePool<CudnnHandle> pool = getPool(deviceId);
        try {
          pool.apply(RefUtil.wrapInterface((RefConsumer<CudnnHandle>) gpu -> {
            try {
              gpu.call(RefUtil.addRef(fn));
            } finally {
              gpu.freeRef();
            }
          }, fn));
        } finally {
          pool.freeRef();
        }
      }
    } finally {
      CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice)
        CudaDevice.setDevice(incumbantDevice);
    }
  }

  /**
   * With device t.
   *
   * @param <T>      the type parameter
   * @param deviceId the device id
   * @param fn       the fn
   * @return the t
   */
  public static <T> T withDevice(int deviceId, @Nonnull RefFunction<CudnnHandle, T> fn) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null) {
        if (threadlocal.getDeviceId() == deviceId) {
          try {
            return fn.apply(threadlocal.addRef());
          } finally {
            RefUtil.freeRef(fn);
          }
        }
      }
      ResourcePool<CudnnHandle> pool = getPool(deviceId);
      try {
        return pool.apply(RefUtil.wrapInterface((RefFunction<CudnnHandle, T>) gpu -> {
          try {
            return gpu.call(RefUtil.addRef(fn));
          } finally {
            gpu.freeRef();
          }
        }, fn));
      } finally {
        pool.freeRef();
      }
    } finally {
      CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice)
        CudaDevice.setDevice(incumbantDevice);
    }
  }

  /**
   * Run.
   *
   * @param fn    the fn
   * @param hints the hints
   */
  public static void run(@Nonnull @RefAware final RefConsumer<CudnnHandle> fn, @Nonnull @RefAware Object... hints) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null) {
        assert isThreadDeviceId(threadlocal.getDeviceId());
        RefUtil.freeRef(hints);
        try {
          fn.accept(threadlocal.addRef());
        } finally {
          RefUtil.freeRef(fn);
        }
      } else {
        int device = chooseDevice(hints);
        ResourcePool<CudnnHandle> pool = getPool(device);
        try {
          pool.apply(RefUtil.wrapInterface((RefConsumer<CudnnHandle>) gpu -> {
            try {
              gpu.call(RefUtil.addRef(fn));
            } finally {
              gpu.freeRef();
            }
          }, fn));
        } finally {
          pool.freeRef();
        }
      }
    } finally {
      CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice)
        CudaDevice.setDevice(incumbantDevice);
    }
  }

  /**
   * Run t.
   *
   * @param <T>   the type parameter
   * @param fn    the fn
   * @param hints the hints
   * @return the t
   */
  public static <T> T run(@Nonnull @RefAware final RefFunction<CudnnHandle, T> fn, @Nonnull @RefAware Object... hints) {
    CudnnHandle threadLocal = CudnnHandle.threadContext.get();
    final Integer incumbentDevice = getThreadDeviceId();
    try {
      if (threadLocal != null) {
        assert CudaDevice.isThreadDeviceId(threadLocal.getDeviceId());
        RefUtil.freeRef(hints);
        try {
          return fn.apply(threadLocal.addRef());
        } finally {
          RefUtil.freeRef(fn);
        }
      }
      int device = chooseDevice(hints);
      assert device >= 0;
      ResourcePool<CudnnHandle> pool = getPool(device);
      try {
        return pool.apply(RefUtil.wrapInterface((RefFunction<CudnnHandle, T>) gpu -> {
          RefUtil.assertAlive(gpu);
          try {
            return gpu.call(RefUtil.addRef(fn));
          } finally {
            gpu.freeRef();
          }
        }, fn));
      } finally {
        pool.freeRef();
      }
    } finally {
      CudnnHandle.threadContext.set(threadLocal);
      if (null != incumbentDevice) {
        CudaDevice.setDevice(incumbentDevice);
      }
    }
  }

  /**
   * Choose device int.
   *
   * @param hints the hints
   * @return the int
   */
  public static int chooseDevice(@Nonnull @RefAware final Object[] hints) {
    RefSet<Integer> devices = RefArrays.stream(hints).map(hint -> {
      try {
        if (hint instanceof Result) {
          TensorList data = ((Result) hint).getData();
          if (data instanceof CudaTensorList) {
            int deviceId = ((CudaTensorList) data).getDeviceId();
            assert deviceId >= 0;
            data.freeRef();
            return deviceId;
          }
          data.freeRef();
        } else if (hint instanceof CudaDeviceResource) {
          int deviceId = ((CudaDeviceResource) hint).getDeviceId();
          //assert deviceId >= 0 : String.format("%s/%d", hint.getClass(), deviceId);
          if (deviceId >= 0)
            return deviceId;
        } else if (hint instanceof Integer) {
          Integer deviceId = (Integer) hint;
          assert deviceId >= 0;
          return deviceId;
        }
      } finally {
        RefUtil.freeRef(hint);
      }
      return null;
    }).filter(x -> x != null).collect(RefCollectors.toSet());
    if (devices.isEmpty()) {
      RefList<String> candidates = RefArrays.stream(CudaSettings.INSTANCE().defaultDevices.split(","))
          .map(x -> x.trim()).filter(x -> !x.isEmpty()).collect(RefCollectors.toList());
      if (candidates.isEmpty()) {
        int deviceId = (int) Math.floor(Math.random() * getCachedDeviceCount());
        assert deviceId >= 0;
        devices.freeRef();
        candidates.freeRef();
        return deviceId;
      } else {
        devices.freeRef();
        String str = candidates.get((int) (Math.random() * candidates.size()));
        candidates.freeRef();
        return Integer.parseInt(str);
      }
    } else {
      Integer deviceId = RefUtil.get(devices.stream().findAny());
      assert deviceId >= 0;
      devices.freeRef();
      return deviceId;
    }
  }

  /**
   * Synchronize.
   *
   * @param time   the time
   * @param device the device
   */
  public static void synchronize(long time, int device) {
    long startTime = RefSystem.nanoTime();
    Long val = syncTimes.get(device);
    if (null == val)
      val = 0L;
    if (val < time) {
      final Long finalVal = val;
      CharSequence caller = CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : "";
      withDevice(device, gpu -> {
        TimedResult<Long> timedResult = TimedResult.time(() -> cudaDeviceSynchronize());
        Long result = timedResult.getResult();
        CudaTensorList.logger.debug(RefString.format("Synchronized %d in %.4f (%.6f -> %.6f -> %.6f) via %s",
            getThreadDeviceId(), timedResult.seconds(), (finalVal - startTime) / 1e9, (time - startTime) / 1e9,
            (result - startTime) / 1e9, caller));
        gpu.freeRef();
        timedResult.freeRef();
      });
    }
  }

  /**
   * Cuda device synchronize long.
   *
   * @return the long
   */
  public static long cudaDeviceSynchronize() {
    long startTime = RefSystem.nanoTime();
    final int result = JCuda.cudaDeviceSynchronize();
    log("cudaDeviceSynchronize", result, new Object[]{});
    cudaDeviceSynchronize_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    handle(result);
    syncTimes.put(getThreadDeviceId(), startTime);
    return startTime;
  }

  /**
   * Gets pool.
   *
   * @param deviceId the device id
   * @return the pool
   */
  @NotNull
  public static ResourcePool<CudnnHandle> getPool(final int deviceId) {
    assert deviceId >= 0;
    ResourcePool<CudnnHandle> pool = handlePools.computeIfAbsent(deviceId, HandlePool::new);
    pool.assertAlive();
    return pool;
  }

  /**
   * Init int.
   *
   * @return the int
   */
  static int init() {
    if (CudaSettings.INSTANCE().disable) {
      CudaDevice.logger.warn("Disabled CudaSystem");
    }
    final int deviceCount = getDeviceCount();
    for (int d = 0; d < deviceCount; d++) {
      initDevice(d);
    }
    return deviceCount;
  }

  /**
   * To map hash map.
   *
   * @param obj the obj
   * @return the hash map
   */
  @Nonnull
  protected static HashMap<CharSequence, CharSequence> toMap(@Nonnull DoubleStatistics obj) {
    @Nonnull
    HashMap<CharSequence, CharSequence> map = new HashMap<>();
    if (0 < obj.getCount()) {
      map.put("stddev", Double.toString(obj.getStandardDeviation()));
      map.put("mean", Double.toString(obj.getAverage()));
      map.put("total", Double.toString(obj.getSum()));
      map.put("max", Double.toString(obj.getMax()));
      map.put("count", Double.toString(obj.getCount()));
    }
    return map;
  }

  /**
   * Render to log char sequence.
   *
   * @param obj the obj
   * @return the char sequence
   */
  protected static CharSequence renderToLog(final Object obj) {
    if (obj instanceof int[]) {
      if (((int[]) obj).length < 10) {
        return RefArrays.toString((int[]) obj);
      }
    }
    if (obj instanceof double[]) {
      if (((double[]) obj).length < 10) {
        return RefArrays.toString((double[]) obj);
      }
    }
    if (obj instanceof float[]) {
      if (((float[]) obj).length < 10) {
        return RefArrays.toString((float[]) obj);
      }
    }
    if (obj instanceof long[]) {
      if (((long[]) obj).length < 10) {
        return RefArrays.toString((long[]) obj);
      }
    }
    return obj.toString();
  }

  private static void initDevice(final int deviceNumber) {
    CudaDevice.setDevice(deviceNumber);
    CudaDevice.logger.info(RefString.format("Device %s - %s", deviceNumber, CudaDevice.getDeviceName(deviceNumber)));
    try {
      //CudaSystem.handle(CudaSystem.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync));
    } catch (Throwable e) {
      CudaDevice.logger.warn("Error initializing GPU", e);
      throw Util.throwException(e);
    }
    for (@Nonnull
        DeviceLimits limit : DeviceLimits.values()) {
      CudaDevice.logger.info(RefString.format("Default Limit %s = %s", limit, limit.get()));
    }
    //DeviceLimits.HeapSize.set(16 * 1024 * 1024 * 1024);
    //DeviceLimits.FifoSize.set(8 * 1024 * 1024);
//    for (@Nonnull
//        DeviceLimits limit : DeviceLimits.values()) {
//      CudaDevice.logger.info(RefString.format("Configured Limit %s = %s", limit, limit.get()));
//    }
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }


  /**
   * The interface Cuda device resource.
   */
  public interface CudaDeviceResource {
    /**
     * Gets device id.
     *
     * @return the device id
     */
    int getDeviceId();
  }

  private static class HandlePool extends ResourcePool<CudnnHandle> {
    private final Integer deviceId;

    /**
     * Instantiates a new Handle pool.
     *
     * @param deviceId the device id
     */
    public HandlePool(Integer deviceId) {
      super(CudaSettings.INSTANCE().handlesPerDevice);
      this.deviceId = deviceId;
    }

    @Nonnull
    @Override
    public CudnnHandle create() {
      return new CudnnHandle(deviceId);
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }
  }
}
