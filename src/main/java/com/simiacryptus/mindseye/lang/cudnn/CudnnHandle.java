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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.lang.ResourcePool;
import com.simiacryptus.mindseye.lang.ReshapedTensorList;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * The type Cudnn handle.
 */
public class CudnnHandle extends CudaDevice {
  /**
   * The Thread context.
   */
  static final RefThreadLocal<CudnnHandle> threadContext = new RefThreadLocal<>();
  private static final ExecutorService cleanupPool = Executors.newFixedThreadPool(1,
      new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The Cleanup native.
   */
  public final LinkedBlockingQueue<CudaResourceBase> cleanupNative = new LinkedBlockingQueue<>();
  /**
   * The Handle.
   */
  @Nullable
  public final cudnnHandle handle;

  /**
   * Instantiates a new Cudnn handle.
   *
   * @param deviceNumber the device number
   */
  CudnnHandle(final int deviceNumber) {
    super(deviceNumber);
    if (0 <= this.deviceId) {
      initThread();
      handle = new cudnnHandle();
      JCudnn.cudnnCreate(handle);
    } else {
      handle = null;
    }
    //cudaSetDevice();
  }

  /**
   * For each.
   *
   * @param fn the fn
   */
  public static void forEach(@Nonnull final RefConsumer<? super CudnnHandle> fn) {
    RefSet<Integer> keys = handlePools.keySet();
    try {
      keys.forEach(device -> {
        ResourcePool<CudnnHandle> pool = getPool(device);
        try {
          pool.apply(cudnnHandle -> {
            cudnnHandle.initThread();
            fn.accept(cudnnHandle);
          });
        } finally {
          pool.freeRef();
        }
      });
    } finally {
      RefUtil.freeRef(fn);
      keys.freeRef();
    }
  }

  /**
   * Cudnn destroy reduce tensor descriptor int.
   *
   * @param obj the obj
   * @return the int
   */
  public static int cudnnDestroyReduceTensorDescriptor(final cudnnReduceTensorDescriptor obj) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyReduceTensorDescriptor(obj);
    cudnnDestroyOpTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyOpTensorDescriptor", result, new Object[]{obj});
    return result;
  }

  /**
   * Get t.
   *
   * @param <T> the type parameter
   * @param obj the obj
   * @return the t
   */
  @org.jetbrains.annotations.Nullable
  @RefAware
  public static <T extends ReferenceCounting> T get(@RefIgnore T obj) {
    if (null == obj) return null;
    else if (obj.isFreed()) return null;
    else return (T) obj.addRef();
  }

  /**
   * Call.
   *
   * @param fn the fn
   */
  public @RefAware
  void call(@Nonnull @RefAware final RefConsumer<CudnnHandle> fn) {
    try {
      executionThread.submit(() -> {
        threadContext.set(CudnnHandle.this.addRef());
        initThread();
        assert isThreadDeviceId(deviceId);
        fn.accept(CudnnHandle.this.addRef());
      }).get();
    } catch (ExecutionException e) {
      throw Util.throwException(e.getCause());
    } catch (Exception e) {
      throw Util.throwException(e);
    } finally {
      RefUtil.freeRef(fn);
      cleanup();
    }
  }

  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  @Nonnull
  public @RefAware
  <T> T call(@Nonnull @RefAware final RefFunction<CudnnHandle, T> fn) {
    try {
      return executionThread.submit(() -> {
        threadContext.set(CudnnHandle.this.addRef());
        initThread();
        assert isThreadDeviceId(deviceId);
        return fn.apply(CudnnHandle.this.addRef());
      }).get();
    } catch (ExecutionException e) {
      throw Util.throwException(e.getCause());
    } catch (Exception e) {
      throw Util.throwException(e);
    } finally {
      RefUtil.freeRef(fn);
      cleanup();
    }
  }

  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  @Nonnull
  public @RefAware
  <T> T call(@Nonnull @RefAware final RefSupplier<T> fn) {
    try {
      return executionThread.submit(() -> {
        threadContext.set(CudnnHandle.this.addRef());
        initThread();
        assert isThreadDeviceId(deviceId);
        return fn.get();
      }).get();
    } catch (ExecutionException e) {
      throw Util.throwException(e.getCause());
    } catch (Exception e) {
      throw Util.throwException(e);
    } finally {
      RefUtil.freeRef(fn);
      cleanup();
    }
  }

  /**
   * Add cuda tensor list.
   *
   * @param left  the left
   * @param right the right
   * @return the cuda tensor list
   */
  @Nonnull
  public CudaTensorList add(@Nonnull final CudaTensorList left, @Nonnull final CudaTensorList right) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    int length = left.length();
    int[] dimensions = right.getDimensions();
    assert dimensions.length <= 3;
    int d2 = dimensions.length < 3 ? 1 : dimensions[2];
    int d1 = dimensions.length < 2 ? 1 : dimensions[1];
    int d0 = dimensions[0];
    Precision precision = right.getPrecision();
    @Nonnull
    CudaTensor rPtr = getTensor(right.addRef(), MemoryType.Device, false);
    right.freeRef();
    @Nonnull
    CudaTensor lPtr = getTensor(left.addRef(), MemoryType.Device, false);
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD,
        precision);
    final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(left.getPrecision(), length, d2, d1,
        d0, d2 * d1 * d0, d1 * d0, d0, 1);
    left.freeRef();
    @Nonnull final CudaMemory outputPtr = allocate((long) outputDescriptor.nStride * precision.size * length,
        MemoryType.Managed.ifEnabled(), true);
    CudaMemory lPtrMemory = lPtr.getMemory(this.addRef());
    CudaMemory rPtrMemory = rPtr.getMemory(this.addRef());
    assert rPtrMemory != null;
    assert lPtrMemory != null;
    cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(), precision.getPointer(0.0),
        outputDescriptor.getPtr(), outputPtr.getPtr());
    lPtrMemory.dirty();
    lPtrMemory.freeRef();
    rPtrMemory.dirty();
    rPtrMemory.freeRef();
    outputPtr.dirty();
    rPtr.freeRef();
    lPtr.freeRef();
    opDescriptor.freeRef();
    return new CudaTensorList(new CudaTensor(outputPtr,
        outputDescriptor, precision), length, dimensions, precision);
  }

  /**
   * Add in place cuda tensor list.
   *
   * @param left  the left
   * @param right the right
   * @return the cuda tensor list
   */
  @Nonnull
  public CudaTensorList addInPlace(@Nullable final CudaTensorList left, @Nullable final TensorList right) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    assert left != null;
    @Nullable final CudaTensor lPtr = getTensor(left.addRef(), left.getPrecision(), MemoryType.Device,
        false);//.moveTo(gpu.getDeviceNumber());
    @Nullable final CudaTensor rPtr = getTensor(right == null ? null : right.addRef(), left.getPrecision(), MemoryType.Device,
        false);//.moveTo(gpu.getDeviceNumber());
    if (null != right)
      right.freeRef();
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    CudaMemory rPtrMemory = rPtr.getMemory(this.addRef());
    CudaMemory lPtrMemory = lPtr.getMemory(this.addRef());
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    assert lPtrMemory != null;
    assert rPtrMemory != null;
    cudnnAddTensor(left.getPrecision().getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
        left.getPrecision().getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr());
    rPtrMemory.dirty();
    lPtrMemory.dirty();
    lPtr.freeRef();
    rPtr.freeRef();
    rPtrMemory.freeRef();
    lPtrMemory.freeRef();
    return left;
  }

  /**
   * Gets tensor.
   *
   * @param data       the data
   * @param precision  the precision
   * @param memoryType the memory type
   * @param dense      the dense
   * @return the tensor
   */
  @Nonnull
  public CudaTensor getTensor(@Nonnull final TensorList data, @Nonnull final Precision precision,
                              @Nonnull final MemoryType memoryType, final boolean dense) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    try {
      data.assertAlive();
      if (data instanceof ReshapedTensorList) {
        ReshapedTensorList reshapedTensorList = (ReshapedTensorList) data.addRef();
        int[] newDims = reshapedTensorList.getDimensions();
        CudaTensor reshapedTensor = getTensor(reshapedTensorList.getInner(), precision, memoryType, true);
        reshapedTensorList.freeRef();
        int channels = newDims.length < 3 ? 1 : newDims[2];
        int height = newDims.length < 2 ? 1 : newDims[1];
        int width = newDims.length < 1 ? 1 : newDims[0];
        CudaTensorDescriptor descriptor = newTensorDescriptor(precision, reshapedTensor.descriptor.batchCount, channels,
            height, width, channels * height * width, height * width, width, 1);
        CudaMemory tensorMemory = reshapedTensor.getMemory(this.addRef(), memoryType);
        reshapedTensor.freeRef();
        CudaTensor temp_53_0002 = new CudaTensor(tensorMemory == null ? null : tensorMemory.addRef(),
            descriptor.addRef(), precision);
        if (null != tensorMemory)
          tensorMemory.freeRef();
        descriptor.freeRef();
        return temp_53_0002;
      }
      if (data instanceof CudaTensorList) {
        CudaTensorList cudaTensorList = (CudaTensorList) data.addRef();
        if (precision == cudaTensorList.getPrecision()) {
          return this.getTensor(cudaTensorList, memoryType, dense);
        } else {
          if (CudaSettings.INSTANCE().verbose) {
            CudaTensorList.logger.warn(RefString.format(
                "Incompatible precision types %s != %s for Tensor %s in GPU at %s, created by %s", precision,
                cudaTensorList.getPrecision(),
                Integer.toHexString(RefSystem.identityHashCode(cudaTensorList.addRef())),
                Util.toString(Util.getStackTrace()).replaceAll("\n", ", "),
                Util.toString(cudaTensorList.createdBy).replaceAll("\n", ", ")));
          } else {
            if (CudaTensorList.logger.isDebugEnabled()) CudaTensorList.logger.debug(RefString.format(
                "Incompatible precision types %s != %s for Tensor %s in GPU at %s, created by %s", precision,
                cudaTensorList.getPrecision(),
                Integer.toHexString(RefSystem.identityHashCode(cudaTensorList.addRef())),
                Util.toString(Util.getStackTrace()).replaceAll("\n", ", "),
                Util.toString(cudaTensorList.createdBy).replaceAll("\n", ", ")));
          }
        }
        cudaTensorList.freeRef();
      }
      final int listLength = data.length();
      if (listLength <= 0) {
        throw new IllegalStateException(RefString.format("listLength = %d", listLength));
      }
      final int elementLength = Tensor.length(data.getDimensions());
      if (elementLength <= 0) {
        throw new IllegalStateException(RefString.format("elementLength = %d", elementLength));
      }
      @Nonnull final CudaMemory ptr = this.allocate((long) elementLength * listLength * precision.size, memoryType, true);
      for (int i = 0; i < listLength; i++) {
        Tensor tensor = data.get(i);
        assert RefArrays.equals(tensor.getDimensions(),
            data.getDimensions()) : RefArrays.toString(tensor.getDimensions()) + " != "
            + RefArrays.toString(data.getDimensions());
        ptr.write(precision, tensor, (long) i * elementLength);
      }
      int[] inputSize = data.getDimensions();
      final int channels = inputSize.length < 3 ? 1 : inputSize[2];
      final int height = inputSize.length < 2 ? 1 : inputSize[1];
      final int width = inputSize.length < 1 ? 1 : inputSize[0];
      return new CudaTensor(ptr, newTensorDescriptor(
          precision,
          data.length(),
          channels,
          height,
          width,
          channels * height * width,
          height * width,
          width,
          1
      ), precision);
    } finally {
      data.freeRef();
    }
  }

  /**
   * Gets tensor.
   *
   * @param data       the data
   * @param memoryType the memory type
   * @param dense      the dense
   * @return the tensor
   */
  @Nonnull
  public CudaTensor getTensor(@Nonnull final CudaTensorList data, @Nonnull final MemoryType memoryType,
                              final boolean dense) {
    final CudaTensor result = getCudaTensor(data.addRef(), memoryType, dense);
    data.setCudaTensor(result.addRef());
    data.freeRef();
    return result;
  }

  /**
   * Dense filter cuda tensor.
   *
   * @param result the result
   * @param dense  the dense
   * @return the cuda tensor
   */
  public CudaTensor denseFilter(CudaTensor result, boolean dense) {
    if (dense || CudaSettings.INSTANCE().allDense) {
      return getDense(result);
    } else {
      return result;
    }
  }

  /**
   * Gets dense.
   *
   * @param result the result
   * @return the dense
   */
  @NotNull
  public CudaTensor getDense(CudaTensor result) {
    CudaTensor dense = result.getDense(this.addRef());
    RefUtil.freeRef(result);
    return dense;
  }

  /**
   * Add and free tensor list.
   *
   * @param precision the precision
   * @param left      the left
   * @param right     the right
   * @return the tensor list
   */
  @Nonnull
  public TensorList addAndFree(@Nonnull final Precision precision, @Nonnull final TensorList left, @Nonnull final TensorList right) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    final int[] dimensions = left.getDimensions();
    assert left.length() == right.length();
    assert Tensor.length(left.getDimensions()) == Tensor.length(right.getDimensions());
    int length = left.length();
    assert length == right.length();
    if (left.currentRefCount() == 1 && left instanceof CudaTensorList) {
      CudaTensor leftGpu = ((CudaTensorList) left).cudaTensor.addRef();
      if (leftGpu.memory.getDeviceId() == getDeviceId()) {
        leftGpu.freeRef();
        CudaTensorList temp_53_0006 = addInPlace(((CudaTensorList) left).addRef(),
            right.addRef());
        left.freeRef();
        right.freeRef();
        return temp_53_0006;
      }
      leftGpu.freeRef();
    }
    if (right.currentRefCount() == 1 && right instanceof CudaTensorList) {
      CudaTensor rightGpu = ((CudaTensorList) right).cudaTensor.addRef();
      if (rightGpu.memory.getDeviceId() == getDeviceId()) {
        rightGpu.freeRef();
        CudaTensorList temp_53_0007 = addInPlace(((CudaTensorList) right).addRef(),
            left.addRef());
        left.freeRef();
        right.freeRef();
        return temp_53_0007;
      }
      rightGpu.freeRef();
    }
    @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD,
        precision);
    final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(precision, length, dimensions[2],
        dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
        dimensions[0], 1);
    @Nullable final CudaTensor lPtr = getTensor(left.addRef(), precision, MemoryType.Device, false);//.moveTo(gpu.getDeviceNumber());
    left.freeRef();
    @Nullable final CudaTensor rPtr = getTensor(right.addRef(), precision, MemoryType.Device, false);//.moveTo(gpu.getDeviceNumber());
    right.freeRef();
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    @Nonnull final CudaMemory outputPtr = allocate(outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
    CudaMemory lPtrMemory = lPtr.getMemory(addRef());
    CudaMemory rPtrMemory = rPtr.getMemory(addRef());
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    assert rPtrMemory != null;
    assert lPtrMemory != null;
    cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(), precision.getPointer(0.0),
        outputDescriptor.getPtr(), outputPtr.getPtr());
    rPtr.freeRef();
    lPtr.freeRef();
    opDescriptor.freeRef();
    lPtrMemory.dirty();
    lPtrMemory.freeRef();
    rPtrMemory.dirty();
    rPtrMemory.freeRef();
    outputPtr.dirty();
    return new CudaTensorList(
        new CudaTensor(outputPtr, outputDescriptor, precision),
        length, dimensions, precision);
  }

  /**
   * Cudnn activation forward int.
   *
   * @param activationDesc the activation desc
   * @param alpha          the alpha
   * @param xDesc          the x desc
   * @param x              the x
   * @param beta           the beta
   * @param yDesc          the y desc
   * @param y              the y
   * @return the int
   */
  public int cudnnActivationForward(final cudnnActivationDescriptor activationDesc, final CudaPointer alpha,
                                    final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta, final cudnnTensorDescriptor yDesc,
                                    final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnActivationForward(this.handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnActivationForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnActivationForward", result, new Object[]{this.addRef(), activationDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn add tensor.
   *
   * @param alpha the alpha
   * @param aDesc the a desc
   * @param A     the a
   * @param beta  the beta
   * @param cDesc the c desc
   * @param C     the c
   */
  public void cudnnAddTensor(final CudaPointer alpha, final cudnnTensorDescriptor aDesc, final CudaPointer A,
                             final CudaPointer beta, final cudnnTensorDescriptor cDesc, final CudaPointer C) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnAddTensor(this.handle, alpha, aDesc, A, beta, cDesc, C);
    cudnnAddTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnAddTensor", result, new Object[]{this.addRef(), alpha, aDesc, A, beta, cDesc, C});
    CudaSystem.handle(result);
  }

  /**
   * Cudnn convolution backward bias.
   *
   * @param alpha  the alpha
   * @param dyDesc the dy desc
   * @param dy     the dy
   * @param beta   the beta
   * @param dbDesc the db desc
   * @param db     the db
   */
  public void cudnnConvolutionBackwardBias(final CudaPointer alpha, final cudnnTensorDescriptor dyDesc,
                                           final CudaPointer dy, final CudaPointer beta, final cudnnTensorDescriptor dbDesc, final CudaPointer db) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardBias(this.handle, alpha, dyDesc, dy, beta, dbDesc, db);
    cudnnConvolutionBackwardBias_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionBackwardBias", result, new Object[]{this.addRef(), alpha, dyDesc, dy, beta, dbDesc, db});
  }

  /**
   * Cudnn convolution backward data int.
   *
   * @param alpha                the alpha
   * @param wDesc                the w desc
   * @param w                    the w
   * @param dyDesc               the dy desc
   * @param dy                   the dy
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param dxDesc               the dx desc
   * @param dx                   the dx
   * @return the int
   */
  public int cudnnConvolutionBackwardData(final CudaPointer alpha, final cudnnFilterDescriptor wDesc,
                                          final CudaPointer w, final cudnnTensorDescriptor dyDesc, final CudaPointer dy,
                                          final cudnnConvolutionDescriptor convDesc, final int algo, final CudaPointer workSpace,
                                          final long workSpaceSizeInBytes, final CudaPointer beta, final cudnnTensorDescriptor dxDesc,
                                          final CudaPointer dx) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardData(this.handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    cudnnConvolutionBackwardData_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionBackwardData", result, new Object[]{this.addRef(), alpha, wDesc, w, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dxDesc, dx});
    return result;
  }

  /**
   * Cudnn convolution backward filter int.
   *
   * @param alpha                the alpha
   * @param xDesc                the x desc
   * @param x                    the x
   * @param dyDesc               the dy desc
   * @param dy                   the dy
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param dwDesc               the dw desc
   * @param dw                   the dw
   * @return the int
   */
  public int cudnnConvolutionBackwardFilter(final CudaPointer alpha, final cudnnTensorDescriptor xDesc,
                                            final CudaPointer x, final cudnnTensorDescriptor dyDesc, final CudaPointer dy,
                                            final cudnnConvolutionDescriptor convDesc, final int algo, final CudaPointer workSpace,
                                            final long workSpaceSizeInBytes, final CudaPointer beta, final cudnnFilterDescriptor dwDesc,
                                            final CudaPointer dw) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardFilter(this.handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    cudnnConvolutionBackwardFilter_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionBackwardFilter", result, new Object[]{this.addRef(), alpha, xDesc, x, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dwDesc, dw});
    return result;
  }

  /**
   * Cudnn convolution forward int.
   *
   * @param alpha                the alpha
   * @param xDesc                the x desc
   * @param x                    the x
   * @param wDesc                the w desc
   * @param w                    the w
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param yDesc                the y desc
   * @param y                    the y
   * @return the int
   */
  public int cudnnConvolutionForward(final CudaPointer alpha, final cudnnTensorDescriptor xDesc, final CudaPointer x,
                                     final cudnnFilterDescriptor wDesc, final CudaPointer w, final cudnnConvolutionDescriptor convDesc, final int algo,
                                     final CudaPointer workSpace, final long workSpaceSizeInBytes, final CudaPointer beta,
                                     final cudnnTensorDescriptor yDesc, final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionForward(this.handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, yDesc, y);
    cudnnConvolutionForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionForward", result, new Object[]{this.addRef(), alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn convolution bias activation forward int.
   *
   * @param alpha                the alpha
   * @param xDesc                the x desc
   * @param x                    the x
   * @param wDesc                the w desc
   * @param w                    the w
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param zDesc                the z desc
   * @param z                    the z
   * @param biasDesc             the bias desc
   * @param bias                 the bias
   * @param activationDesc       the activation desc
   * @param yDesc                the y desc
   * @param y                    the y
   * @return the int
   */
  public int cudnnConvolutionBiasActivationForward(final CudaPointer alpha, final cudnnTensorDescriptor xDesc,
                                                   final CudaPointer x, final cudnnFilterDescriptor wDesc, final CudaPointer w,
                                                   final cudnnConvolutionDescriptor convDesc, final int algo, final CudaPointer workSpace,
                                                   final long workSpaceSizeInBytes, final CudaPointer beta,

                                                   final cudnnTensorDescriptor zDesc, final CudaPointer z, final cudnnTensorDescriptor biasDesc,
                                                   final CudaPointer bias, final cudnnActivationDescriptor activationDesc,

                                                   final cudnnTensorDescriptor yDesc, final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionBiasActivationForward(this.handle, alpha, xDesc, x, wDesc, w, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
    cudnnConvolutionBiasActivationForward_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionBiasActivationForward", result, new Object[]{this.addRef(), alpha, xDesc, x, wDesc, w, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, zDesc, z, biasDesc, bias, activationDesc, yDesc, y});
    return result;
  }

  /**
   * Cudnn op tensor int.
   *
   * @param opTensorDesc the op tensor desc
   * @param alpha1       the alpha 1
   * @param aDesc        the a desc
   * @param A            the a
   * @param alpha2       the alpha 2
   * @param bDesc        the b desc
   * @param B            the b
   * @param beta         the beta
   * @param cDesc        the c desc
   * @param C            the c
   * @return the int
   */
  public int cudnnOpTensor(final cudnnOpTensorDescriptor opTensorDesc, final CudaPointer alpha1,
                           final cudnnTensorDescriptor aDesc, final CudaPointer A, final CudaPointer alpha2,
                           final cudnnTensorDescriptor bDesc, final CudaPointer B, final CudaPointer beta, final cudnnTensorDescriptor cDesc,
                           final CudaPointer C) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnOpTensor(this.handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc,
        C);
    cudnnOpTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnOpTensor", result,
        new Object[]{this.addRef(), opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C});
    return result;
  }

  /**
   * Cudnn reduce tensor int.
   *
   * @param reduceTensorDesc     the reduce tensor desc
   * @param indices              the indices
   * @param indicesSizeInBytes   the indices size in bytes
   * @param workspace            the workspace
   * @param workspaceSizeInBytes the workspace size in bytes
   * @param alpha                the alpha
   * @param aDesc                the a desc
   * @param A                    the a
   * @param beta                 the beta
   * @param cDesc                the c desc
   * @param C                    the c
   * @return the int
   */
  public int cudnnReduceTensor(cudnnReduceTensorDescriptor reduceTensorDesc, Pointer indices, long indicesSizeInBytes,
                               Pointer workspace, long workspaceSizeInBytes, Pointer alpha, cudnnTensorDescriptor aDesc, Pointer A, Pointer beta,
                               cudnnTensorDescriptor cDesc, Pointer C) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnReduceTensor(this.handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
        workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
    cudnnReduceTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnReduceTensor", result, new Object[]{this.addRef(), reduceTensorDesc, indices, indicesSizeInBytes, workspace,
        workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C});
    return result;
  }

  /**
   * Cudnn pooling backward int.
   *
   * @param poolingDesc the pooling desc
   * @param alpha       the alpha
   * @param yDesc       the y desc
   * @param y           the y
   * @param dyDesc      the dy desc
   * @param dy          the dy
   * @param xDesc       the x desc
   * @param x           the x
   * @param beta        the beta
   * @param dxDesc      the dx desc
   * @param dx          the dx
   * @return the int
   */
  public int cudnnPoolingBackward(final cudnnPoolingDescriptor poolingDesc, final CudaPointer alpha,
                                  final cudnnTensorDescriptor yDesc, final CudaPointer y, final cudnnTensorDescriptor dyDesc, final CudaPointer dy,
                                  final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta,
                                  final cudnnTensorDescriptor dxDesc, final CudaPointer dx) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnPoolingBackward(this.handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x,
        beta, dxDesc, dx);
    cudnnPoolingBackward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnPoolingBackward", result,
        new Object[]{this.addRef(), poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }

  /**
   * Cudnn pooling forward int.
   *
   * @param poolingDesc the pooling desc
   * @param alpha       the alpha
   * @param xDesc       the x desc
   * @param x           the x
   * @param beta        the beta
   * @param yDesc       the y desc
   * @param y           the y
   * @return the int
   */
  public int cudnnPoolingForward(final cudnnPoolingDescriptor poolingDesc, final CudaPointer alpha,
                                 final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta, final cudnnTensorDescriptor yDesc,
                                 final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnPoolingForward(this.handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnPoolingForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnPoolingForward", result, new Object[]{this.addRef(), poolingDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn lrn cross channel forward int.
   *
   * @param normDesc the norm desc
   * @param lrnMode  the lrn mode
   * @param alpha    the alpha
   * @param xDesc    the x desc
   * @param x        the x
   * @param beta     the beta
   * @param yDesc    the y desc
   * @param y        the y
   * @return the int
   */
  public int cudnnLRNCrossChannelForward(final cudnnLRNDescriptor normDesc, final int lrnMode, final CudaPointer alpha,
                                         final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta, final cudnnTensorDescriptor yDesc,
                                         final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnLRNCrossChannelForward(this.handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc,
        y);
    cudnnLRNCrossChannelForward_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnLRNCrossChannelForward", result,
        new Object[]{this.addRef(), normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn lrn cross channel backward int.
   *
   * @param normDesc the norm desc
   * @param lrnMode  the lrn mode
   * @param alpha    the alpha
   * @param yDesc    the y desc
   * @param y        the y
   * @param dyDesc   the dy desc
   * @param dy       the dy
   * @param xDesc    the x desc
   * @param x        the x
   * @param beta     the beta
   * @param dxDesc   the dx desc
   * @param dx       the dx
   * @return the int
   */
  public int cudnnLRNCrossChannelBackward(cudnnLRNDescriptor normDesc, int lrnMode, Pointer alpha,
                                          cudnnTensorDescriptor yDesc, Pointer y, cudnnTensorDescriptor dyDesc, Pointer dy, cudnnTensorDescriptor xDesc,
                                          Pointer x, Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();

    final int result = JCudnn.cudnnLRNCrossChannelBackward(this.handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy,
        xDesc, x, beta, dxDesc, dx);
    cudnnLRNCrossChannelBackward_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnLRNCrossChannelBackward", result,
        new Object[]{this.handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }

  /**
   * Cudnn set lrn descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @param n           the n
   * @param alpha       the alpha
   * @param beta        the beta
   * @param k           the k
   * @return the int
   */
  public int cudnnSetLRNDescriptor(final cudnnLRNDescriptor poolingDesc, final int n, final double alpha,
                                   final double beta, final double k) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSetLRNDescriptor(poolingDesc, n, alpha, beta, k);
    cudnnSetLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetLRNDescriptor", result, new Object[]{poolingDesc, n, alpha, beta, k});
    return result;
  }

  /**
   * Cudnn create lrn descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @return the int
   */
  public int cudnnCreateLRNDescriptor(final cudnnLRNDescriptor poolingDesc) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnCreateLRNDescriptor(poolingDesc);
    cudnnCreateLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnCreateLRNDescriptor", result, new Object[]{poolingDesc});
    return result;
  }

  /**
   * Cudnn destroy lrn descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @return the int
   */
  public int cudnnDestroyLRNDescriptor(final cudnnLRNDescriptor poolingDesc) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyLRNDescriptor(poolingDesc);
    cudnnDestroyLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyLRNDescriptor", result, new Object[]{poolingDesc});
    return result;
  }

  /**
   * Allocate backward filter workspace cuda memory.
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @param minSize       the min size
   * @return the cuda memory
   */
  @Nonnull
  public CudaMemory allocateBackwardFilterWorkspace(final cudnnTensorDescriptor srcTensorDesc,
                                                    final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc,
                                                    final cudnnTensorDescriptor dstTensorDesc, final int algorithm, final long minSize) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, srcTensorDesc, dstTensorDesc,
        convDesc, filterDesc, algorithm, sizeInBytesArray);
    allocateBackwardFilterWorkspace_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardFilterWorkspaceSize", result,
        new Object[]{this.addRef(), srcTensorDesc, dstTensorDesc, convDesc, filterDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

  /**
   * Allocate forward workspace cuda memory.
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @param minSize       the min size
   * @return the cuda memory
   */
  @Nonnull
  public CudaMemory allocateForwardWorkspace(final cudnnTensorDescriptor srcTensorDesc,
                                             final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc,
                                             final cudnnTensorDescriptor dstTensorDesc, final int algorithm, final long minSize) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(handle, srcTensorDesc, filterDesc, convDesc,
        dstTensorDesc, algorithm, sizeInBytesArray);
    allocateForwardWorkspace_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionForwardWorkspaceSize", result,
        new Object[]{this.addRef(), srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return this.allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

  /**
   * Gets backward data algorithm.
   *
   * @param dyDesc             the dy desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param dxDesc             the dx desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the backward data algorithm
   */
  public int getBackwardDataAlgorithm(final cudnnTensorDescriptor dyDesc, final cudnnFilterDescriptor filterDesc,
                                      final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dxDesc, final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    cudnnConvolutionBwdDataAlgoPerf[] perf = {new cudnnConvolutionBwdDataAlgoPerf()};
    final int result = JCudnn.cudnnFindConvolutionBackwardDataAlgorithm(handle, filterDesc, dyDesc, convDesc, dxDesc,
        1, algoArray, perf);
    getBackwardDataAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardDataAlgorithm", result, new Object[]{this.addRef(), filterDesc, dyDesc, convDesc, dxDesc,
            1, algoArray, perf});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  /**
   * Gets backward filter algorithm.
   *
   * @param inputDesc          the input desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param outputDesc         the output desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the backward filter algorithm
   */
  public int getBackwardFilterAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc,
                                        final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc,
                                        final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    cudnnConvolutionBwdFilterAlgoPerf[] perfResults = { new cudnnConvolutionBwdFilterAlgoPerf() };
    final int result = JCudnn.cudnnFindConvolutionBackwardFilterAlgorithm(handle, inputDesc, outputDesc, convDesc,
        filterDesc, 1, algoArray, perfResults);
    getBackwardFilterAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardFilterAlgorithm", result,
        new Object[]{this.addRef(), inputDesc, outputDesc, convDesc, filterDesc,
                1, algoArray, perfResults});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  /**
   * Gets forward algorithm.
   *
   * @param srcTensorDesc      the src tensor desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param dstTensorDesc      the dst tensor desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the forward algorithm
   */
  public int getForwardAlgorithm(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc,
                                 final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc,
                                 final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    cudnnConvolutionFwdAlgoPerf[] perfs = { new cudnnConvolutionFwdAlgoPerf() };
    final int result = JCudnn.cudnnFindConvolutionForwardAlgorithm(handle, srcTensorDesc, filterDesc, convDesc,
        dstTensorDesc, 1, algoArray, perfs);
    getForwardAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionForwardAlgorithm", result,
        new Object[]{this.addRef(), srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 1, algoArray, perfs});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  /**
   * Cudnn activation backward int.
   *
   * @param activationDesc the activation desc
   * @param alpha          the alpha
   * @param yDesc          the y desc
   * @param y              the y
   * @param dyDesc         the dy desc
   * @param dy             the dy
   * @param xDesc          the x desc
   * @param x              the x
   * @param beta           the beta
   * @param dxDesc         the dx desc
   * @param dx             the dx
   * @return the int
   */
  public int cudnnActivationBackward(final cudnnActivationDescriptor activationDesc, final CudaPointer alpha,
                                     final cudnnTensorDescriptor yDesc, final CudaPointer y, final cudnnTensorDescriptor dyDesc, final CudaPointer dy,
                                     final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta,
                                     final cudnnTensorDescriptor dxDesc, final CudaPointer dx) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnActivationBackward(this.handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc,
        x, beta, dxDesc, dx);
    cudnnActivationBackward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnActivationBackward", result,
        new Object[]{this.addRef(), activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }

  /**
   * Softmax functions: All of the form "output = alphaList * Op(inputs) + beta * output"
   *
   * @param algo  the algo
   * @param mode  the mode
   * @param alpha the alpha
   * @param xDesc the x desc
   * @param x     the x
   * @param beta  the beta
   * @param yDesc the y desc
   * @param y     the y
   * @return the int
   */
  public int cudnnSoftmaxForward(int algo, int mode, CudaPointer alpha, cudnnTensorDescriptor xDesc, CudaPointer x,
                                 CudaPointer beta, cudnnTensorDescriptor yDesc, CudaPointer y) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSoftmaxForward(this.handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
    cudnnSoftmaxForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSoftmaxForward", result, new Object[]{this.addRef(), algo, mode, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn softmax backward int.
   *
   * @param algo   the algo
   * @param mode   the mode
   * @param alpha  the alpha
   * @param yDesc  the y desc
   * @param y      the y
   * @param dyDesc the dy desc
   * @param dy     the dy
   * @param beta   the beta
   * @param dxDesc the dx desc
   * @param dx     the dx
   * @return the int
   */
  public int cudnnSoftmaxBackward(int algo, int mode, CudaPointer alpha, cudnnTensorDescriptor yDesc, CudaPointer y,
                                  cudnnTensorDescriptor dyDesc, CudaPointer dy, CudaPointer beta, cudnnTensorDescriptor dxDesc, CudaPointer dx) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSoftmaxBackward(this.handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc,
        dx);
    cudnnSoftmaxBackward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSoftmaxBackward", result,
        new Object[]{this.addRef(), algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx});
    return result;
  }

  /**
   * Cudnn transform tensor int.
   *
   * @param alpha the alpha
   * @param xDesc the x desc
   * @param x     the x
   * @param beta  the beta
   * @param yDesc the y desc
   * @param y     the y
   * @return the int
   */
  public int cudnnTransformTensor(final CudaPointer alpha, final cudnnTensorDescriptor xDesc, final CudaPointer x,
                                  final CudaPointer beta, final cudnnTensorDescriptor yDesc, final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnTransformTensor(this.handle, alpha, xDesc, x, beta, yDesc, y);
    cudnnTransformTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnTransformTensor", result, new Object[]{this.addRef(), alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  /**
   * Cudnn set tensor int.
   *
   * @param yDesc    the y desc
   * @param y        the y
   * @param valuePtr the value ptr
   * @return the int
   */
  public int cudnnSetTensor(cudnnTensorDescriptor yDesc, CudaPointer y, CudaPointer valuePtr) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSetTensor(this.handle, yDesc, y, valuePtr);
    cudnnSetTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetTensor", result, new Object[]{this.addRef(), yDesc, y, valuePtr});
    return result;
  }

  /**
   * Allocate backward data workspace cuda memory.
   *
   * @param dxDesc     the dx desc
   * @param filterDesc the filter desc
   * @param convDesc   the conv desc
   * @param dyDesc     the dy desc
   * @param algorithm  the algorithm
   * @param minSize    the min size
   * @return the cuda memory
   */
  @Nonnull
  public CudaMemory allocateBackwardDataWorkspace(final cudnnTensorDescriptor dxDesc,
                                                  final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc,
                                                  final cudnnTensorDescriptor dyDesc, final int algorithm, final long minSize) {
    long size;
    try {
      assert CudaDevice.isThreadDeviceId(getDeviceId());
      long startTime = RefSystem.nanoTime();
      @Nonnull final long sizeInBytesArray[] = {0};
      final int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filterDesc, dyDesc, convDesc,
          dxDesc, algorithm, sizeInBytesArray);
      allocateBackwardDataWorkspace_execution
          .accept((RefSystem.nanoTime() - startTime) / 1e9);
      log("cudnnGetConvolutionBackwardDataWorkspaceSize", result,
          new Object[]{this.addRef(), filterDesc, dyDesc, convDesc, dxDesc, algorithm, sizeInBytesArray});
      CudaSystem.handle(result);
      size = sizeInBytesArray[0];
    } catch (Throwable e) {
      logger.info("Error in allocateBackwardDataWorkspace", e);
      size = 0;
    }
    return this.allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

  /**
   * Cudnn create reduce tensor descriptor.
   *
   * @param reduceTensorDesc the reduce tensor desc
   */
  public void cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor reduceTensorDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    cudnnCreateReduceTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnCreateReduceTensorDescriptor", result, new Object[]{reduceTensorDesc});
  }

  /**
   * Cudnn set reduce tensor descriptor.
   *
   * @param reduceTensorDesc        the reduce tensor desc
   * @param reduceTensorOp          the reduce tensor op
   * @param reduceTensorCompType    the reduce tensor comp type
   * @param reduceTensorNanOpt      the reduce tensor nan opt
   * @param reduceTensorIndices     the reduce tensor indices
   * @param reduceTensorIndicesType the reduce tensor indices type
   */
  public void cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor reduceTensorDesc, int reduceTensorOp,
                                             int reduceTensorCompType, int reduceTensorNanOpt, int reduceTensorIndices, int reduceTensorIndicesType) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
        reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    cudnnSetReduceTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnCreateReduceTensorDescriptor", result, new Object[]{reduceTensorDesc});
  }

  @Nonnull
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceId + "; " + deviceName + "}@"
        + Long.toHexString(RefSystem.identityHashCode(this.addRef()));
  }

  /**
   * Cudnn create reduce tensor descriptor cuda resource.
   *
   * @param reduceTensorOp          the reduce tensor op
   * @param reduceTensorCompType    the reduce tensor comp type
   * @param reduceTensorNanOpt      the reduce tensor nan opt
   * @param reduceTensorIndices     the reduce tensor indices
   * @param reduceTensorIndicesType the reduce tensor indices type
   * @return the cuda resource
   */
  @Nonnull
  public CudaResource<cudnnReduceTensorDescriptor> cudnnCreateReduceTensorDescriptor(int reduceTensorOp,
                                                                                     int reduceTensorCompType, int reduceTensorNanOpt, int reduceTensorIndices, int reduceTensorIndicesType) {
    cudnnReduceTensorDescriptor reduceTensorDesc = new cudnnReduceTensorDescriptor();
    cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType);
    return new CudaResource<cudnnReduceTensorDescriptor>(reduceTensorDesc,
        obj -> cudnnDestroyReduceTensorDescriptor(obj), getDeviceId());
  }

  @Nonnull
  @Override
  public CudnnHandle addRef() {
    return (CudnnHandle) super.addRef();
  }

  @Override
  public void _free() {
    super._free();
    final int result = JCudnn.cudnnDestroy(handle);
    log("cudnnDestroy", result, new Object[]{handle});
    CudaSystem.handle(result);
  }

  /**
   * Cleanup.
   */
  protected void cleanup() {
    ArrayList<CudaResourceBase> objsToFree = new ArrayList<>();
    cleanupNative.drainTo(objsToFree);
    if (!objsToFree.isEmpty()) {
      DeviceMetrics deviceMetrics = CudaMemory.METRICS.get(deviceId);
      if (deviceMetrics.load() < CudaSettings.INSTANCE().asyncFreeLoadThreshold) {
        cleanupPool.submit(() -> {
          freeAll(objsToFree);
        });
      } else {
        freeAll(objsToFree);
      }
    }

  }

  private CudaTensor getCudaTensor(@Nonnull CudaTensorList data, @Nonnull MemoryType memoryType, boolean dense) {
    final CudaTensor gpuCopy;
    final TensorArray heapCopy;
    synchronized (data) {
      gpuCopy = get(data.cudaTensor);
      heapCopy = get(data.heapCopy);
    }
    try {
      assert CudaDevice.isThreadDeviceId(getDeviceId());
      if (gpuCopy == null) {
        if (heapCopy != null) {
          return denseFilter(getTensor(heapCopy.addRef(), data.getPrecision(), memoryType, dense), dense);
        } else {
          throw new IllegalArgumentException();
        }
      } else {
        return denseFilter(gpuCopy.addRef(), dense);
      }
    } finally {
      RefUtil.freeRef(gpuCopy);
      RefUtil.freeRef(heapCopy);
      data.freeRef();
    }
  }

  private void freeAll(List<CudaResourceBase> objsToFree) {
    if (CudaSettings.INSTANCE().syncBeforeFree)
      synchronize(RefSystem.nanoTime(), deviceId);
    objsToFree.forEach(CudaResourceBase::release);
    threadContext.set(null);
  }

}
