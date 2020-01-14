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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import jcuda.Pointer;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CudnnHandle extends CudaDevice {
  static final ThreadLocal<CudnnHandle> threadContext = new ThreadLocal<>();
  private static final ExecutorService cleanupPool = Executors.newFixedThreadPool(1,
      new ThreadFactoryBuilder().setDaemon(true).build());
  public final RefLinkedBlockingQueue<CudaResourceBase> cleanupNative = new RefLinkedBlockingQueue<>();
  @Nullable
  public final cudnnHandle handle;

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

  public static void forEach(@Nonnull final RefConsumer<? super CudnnHandle> fn) {
    RefSet<Integer> temp_53_0008 = handlePools.keySet();
    temp_53_0008.forEach(device -> {
      ResourcePool<CudnnHandle> temp_53_0009 = getPool(device);
      temp_53_0009.apply(x -> {
        x.initThread();
        fn.accept(x);
      });
      temp_53_0009.freeRef();
    });
    temp_53_0008.freeRef();
  }

  public static int cudnnDestroyReduceTensorDescriptor(final cudnnReduceTensorDescriptor obj) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyReduceTensorDescriptor(obj);
    cudnnDestroyOpTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyOpTensorDescriptor", result, new Object[]{obj});
    return result;
  }

  @Nonnull
  public @RefAware
  <T> RefSupplier<T> wrap(@Nonnull @RefAware final RefSupplier<T> fn) {
    return RefUtil.wrapInterface(() -> {
      try {
        return executionThread.submit(() -> {
          CudnnHandle.threadContext.set(CudnnHandle.this);
          initThread();
          assert isThreadDeviceId(deviceId);
          return fn.get();
        }).get();
      } catch (ExecutionException e) {
        throw new RuntimeException(e.getCause());
      } catch (RuntimeException e) {
        throw e;
      } catch (Exception e) {
        throw new RuntimeException(e);
      } finally {
        cleanup();
      }
    }, fn);
  }

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
    @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(left.getPrecision(), length, d2, d1,
        d0, d2 * d1 * d0, d1 * d0, d0, 1);
    left.freeRef();
    @Nonnull final CudaMemory outputPtr = allocate((long) outputDescriptor.nStride * precision.size * length,
        MemoryType.Managed.ifEnabled(), true);
    CudaMemory lPtrMemory = lPtr.getMemory(this);
    CudaMemory rPtrMemory = rPtr.getMemory(this);
    assert rPtrMemory != null;
    assert lPtrMemory != null;
    cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(), precision.getPointer(0.0),
        outputDescriptor.getPtr(), outputPtr.getPtr());
    RefUtil.freeRef(lPtrMemory.dirty());
    lPtrMemory.freeRef();
    RefUtil.freeRef(rPtrMemory.dirty());
    rPtrMemory.freeRef();
    RefUtil.freeRef(outputPtr.dirty());
    rPtr.freeRef();
    lPtr.freeRef();
    opDescriptor.freeRef();
    CudaTensorList temp_53_0001 = new CudaTensorList(new CudaTensor(outputPtr,
        outputDescriptor, precision), length, dimensions, precision);
    return temp_53_0001;
  }

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
    CudaMemory rPtrMemory = rPtr.getMemory(this);
    CudaMemory lPtrMemory = lPtr.getMemory(this);
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    assert lPtrMemory != null;
    assert rPtrMemory != null;
    cudnnAddTensor(left.getPrecision().getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
        left.getPrecision().getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr());
    RefUtil.freeRef(rPtrMemory.dirty());
    RefUtil.freeRef(lPtrMemory.dirty());
    lPtr.freeRef();
    rPtr.freeRef();
    rPtrMemory.freeRef();
    lPtrMemory.freeRef();
    return left;
  }

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
        CudaMemory tensorMemory = reshapedTensor.getMemory(this, memoryType);
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
          CudaTensor temp_53_0003 = this.getTensor(cudaTensorList.addRef(), memoryType,
              dense);
          cudaTensorList.freeRef();
          return temp_53_0003;
        } else {
          String msg = RefString.format(
              "Incompatible precision types %s != %s for Tensor %s in GPU at %s, created by %s", precision,
              cudaTensorList.getPrecision(),
              Integer.toHexString(RefSystem.identityHashCode(cudaTensorList)),
              Util.toString(Util.getStackTrace()).replaceAll("\n", ", "),
              Util.toString(cudaTensorList.createdBy).replaceAll("\n", ", "));
          if (CudaSettings.INSTANCE().verbose) {
            CudaTensorList.logger.warn(msg);
          } else {
            CudaTensorList.logger.debug(msg);
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
        double[] tensorData = tensor.getData();
        tensor.freeRef();
        RefUtil.freeRef(ptr.write(precision, tensorData, (long) i * elementLength));
      }
      int[] inputSize = data.getDimensions();
      final int channels = inputSize.length < 3 ? 1 : inputSize[2];
      final int height = inputSize.length < 2 ? 1 : inputSize[1];
      final int width = inputSize.length < 1 ? 1 : inputSize[0];
      @Nonnull final CudaDevice.CudaTensorDescriptor descriptor = newTensorDescriptor(precision, data.length(), channels, height,
          width, channels * height * width, height * width, width, 1);
      CudaTensor temp_53_0004 = new CudaTensor(ptr, descriptor,
          precision);
      return temp_53_0004;
    } finally {
      data.freeRef();
    }
  }

  @Nonnull
  public CudaTensor getTensor(@Nonnull final CudaTensorList data, @Nonnull final MemoryType memoryType,
                              final boolean dense) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    final CudaTensor gpuCopy;
    final TensorArray heapCopy;
    synchronized (data) {
      gpuCopy = data.gpuCopy.addRef();
      heapCopy = data.heapCopy.addRef();
    }
    CudaTensor result;
    if (gpuCopy.isFinalized() && !heapCopy.isFinalized()) {
      result = getTensor(heapCopy.addRef(), data.getPrecision(), memoryType, dense);
    } else {
      result = gpuCopy.addRef();
    }
    if (dense || CudaSettings.INSTANCE().allDense)
      result = result.getDense(this);
    synchronized (data) {
      if (result != data.gpuCopy) {
        data.gpuCopy = result.addRef();
      }
    }
    gpuCopy.freeRef();
    heapCopy.freeRef();
    data.freeRef();
    return result;
  }

  @Nonnull
  public TensorList addAndFree(@Nonnull final Precision precision, @Nonnull final TensorList left, @Nonnull final TensorList right) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    final int[] dimensions = left.getDimensions();
    assert left.length() == right.length();
    assert Tensor.length(left.getDimensions()) == Tensor.length(right.getDimensions());
    int length = left.length();
    assert length == right.length();
    if (left.currentRefCount() == 1 && left instanceof CudaTensorList) {
      CudaTensor leftGpu = ((CudaTensorList) left).gpuCopy.addRef();
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
      CudaTensor rightGpu = ((CudaTensorList) right).gpuCopy.addRef();
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
    @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(precision, length, dimensions[2],
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
    CudaMemory lPtrMemory = lPtr.getMemory(this);
    CudaMemory rPtrMemory = rPtr.getMemory(this);
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    assert rPtrMemory != null;
    assert lPtrMemory != null;
    cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(), precision.getPointer(0.0),
        outputDescriptor.getPtr(), outputPtr.getPtr());
    rPtr.freeRef();
    lPtr.freeRef();
    opDescriptor.freeRef();
    RefUtil.freeRef(lPtrMemory.dirty());
    lPtrMemory.freeRef();
    RefUtil.freeRef(rPtrMemory.dirty());
    rPtrMemory.freeRef();
    RefUtil.freeRef(outputPtr.dirty());
    CudaTensorList temp_53_0005 = new CudaTensorList(new CudaTensor(outputPtr,
        outputDescriptor, precision), length, dimensions, precision);
    return temp_53_0005;
  }

  public int cudnnActivationForward(final cudnnActivationDescriptor activationDesc, final CudaPointer alpha,
                                    final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta, final cudnnTensorDescriptor yDesc,
                                    final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnActivationForward(this.handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnActivationForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnActivationForward", result, new Object[]{this, activationDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  public void cudnnAddTensor(final CudaPointer alpha, final cudnnTensorDescriptor aDesc, final CudaPointer A,
                             final CudaPointer beta, final cudnnTensorDescriptor cDesc, final CudaPointer C) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnAddTensor(this.handle, alpha, aDesc, A, beta, cDesc, C);
    cudnnAddTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnAddTensor", result, new Object[]{this, alpha, aDesc, A, beta, cDesc, C});
    CudaSystem.handle(result);
  }

  public void cudnnConvolutionBackwardBias(final CudaPointer alpha, final cudnnTensorDescriptor dyDesc,
                                           final CudaPointer dy, final CudaPointer beta, final cudnnTensorDescriptor dbDesc, final CudaPointer db) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardBias(this.handle, alpha, dyDesc, dy, beta, dbDesc, db);
    cudnnConvolutionBackwardBias_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionBackwardBias", result, new Object[]{this, alpha, dyDesc, dy, beta, dbDesc, db});
  }

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
    log("cudnnConvolutionBackwardData", result, new Object[]{this, alpha, wDesc, w, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dxDesc, dx});
    return result;
  }

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
    log("cudnnConvolutionBackwardFilter", result, new Object[]{this, alpha, xDesc, x, dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dwDesc, dw});
    return result;
  }

  public int cudnnConvolutionForward(final CudaPointer alpha, final cudnnTensorDescriptor xDesc, final CudaPointer x,
                                     final cudnnFilterDescriptor wDesc, final CudaPointer w, final cudnnConvolutionDescriptor convDesc, final int algo,
                                     final CudaPointer workSpace, final long workSpaceSizeInBytes, final CudaPointer beta,
                                     final cudnnTensorDescriptor yDesc, final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnConvolutionForward(this.handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, yDesc, y);
    cudnnConvolutionForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnConvolutionForward", result, new Object[]{this, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, yDesc, y});
    return result;
  }

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
    log("cudnnConvolutionBiasActivationForward", result, new Object[]{this, alpha, xDesc, x, wDesc, w, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, zDesc, z, biasDesc, bias, activationDesc, yDesc, y});
    return result;
  }

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
        new Object[]{this, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C});
    return result;
  }

  public int cudnnReduceTensor(cudnnReduceTensorDescriptor reduceTensorDesc, Pointer indices, long indicesSizeInBytes,
                               Pointer workspace, long workspaceSizeInBytes, Pointer alpha, cudnnTensorDescriptor aDesc, Pointer A, Pointer beta,
                               cudnnTensorDescriptor cDesc, Pointer C) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnReduceTensor(this.handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
        workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
    cudnnReduceTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnReduceTensor", result, new Object[]{this, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
        workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C});
    return result;
  }

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
        new Object[]{this, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }

  public int cudnnPoolingForward(final cudnnPoolingDescriptor poolingDesc, final CudaPointer alpha,
                                 final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta, final cudnnTensorDescriptor yDesc,
                                 final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnPoolingForward(this.handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnPoolingForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnPoolingForward", result, new Object[]{this, poolingDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

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
        new Object[]{this, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

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

  public int cudnnSetLRNDescriptor(final cudnnLRNDescriptor poolingDesc, final int n, final double alpha,
                                   final double beta, final double k) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSetLRNDescriptor(poolingDesc, n, alpha, beta, k);
    cudnnSetLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetLRNDescriptor", result, new Object[]{poolingDesc, n, alpha, beta, k});
    return result;
  }

  public int cudnnCreateLRNDescriptor(final cudnnLRNDescriptor poolingDesc) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnCreateLRNDescriptor(poolingDesc);
    cudnnCreateLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnCreateLRNDescriptor", result, new Object[]{poolingDesc});
    return result;
  }

  public int cudnnDestroyLRNDescriptor(final cudnnLRNDescriptor poolingDesc) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnDestroyLRNDescriptor(poolingDesc);
    cudnnDestroyLRNDescriptor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnDestroyLRNDescriptor", result, new Object[]{poolingDesc});
    return result;
  }

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
        new Object[]{this, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

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
        new Object[]{this, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return this.allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

  public int getBackwardDataAlgorithm(final cudnnTensorDescriptor dyDesc, final cudnnFilterDescriptor filterDesc,
                                      final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dxDesc, final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(handle, filterDesc, dyDesc, convDesc, dxDesc,
        cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, memoryLimitInBytes, algoArray);
    getBackwardDataAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardDataAlgorithm", result, new Object[]{this, filterDesc, dyDesc, convDesc, dxDesc,
        cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, memoryLimitInBytes, algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  public int getBackwardFilterAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc,
                                        final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc,
                                        final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(handle, inputDesc, outputDesc, convDesc,
        filterDesc, cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, memoryLimitInBytes,
        algoArray);
    getBackwardFilterAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardFilterAlgorithm", result,
        new Object[]{this, inputDesc, outputDesc, convDesc, filterDesc,
            cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, memoryLimitInBytes,
            algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  public int getForwardAlgorithm(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc,
                                 final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc,
                                 final long memoryLimitInBytes) {
    long startTime = RefSystem.nanoTime();
    @Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(handle, srcTensorDesc, filterDesc, convDesc,
        dstTensorDesc, cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memoryLimitInBytes,
        algoArray);
    getForwardAlgorithm_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionForwardAlgorithm", result,
        new Object[]{this, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
            cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memoryLimitInBytes, algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }

  public int cudnnActivationBackward(final cudnnActivationDescriptor activationDesc, final CudaPointer alpha,
                                     final cudnnTensorDescriptor yDesc, final CudaPointer y, final cudnnTensorDescriptor dyDesc, final CudaPointer dy,
                                     final cudnnTensorDescriptor xDesc, final CudaPointer x, final CudaPointer beta,
                                     final cudnnTensorDescriptor dxDesc, final CudaPointer dx) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnActivationBackward(this.handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc,
        x, beta, dxDesc, dx);
    cudnnActivationBackward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnActivationBackward", result,
        new Object[]{this, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }

  /**
   * Softmax functions: All of the form "output = alphaList * Op(inputs) + beta * output"
   */
  public int cudnnSoftmaxForward(int algo, int mode, CudaPointer alpha, cudnnTensorDescriptor xDesc, CudaPointer x,
                                 CudaPointer beta, cudnnTensorDescriptor yDesc, CudaPointer y) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSoftmaxForward(this.handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
    cudnnSoftmaxForward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSoftmaxForward", result, new Object[]{this, algo, mode, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  public int cudnnSoftmaxBackward(int algo, int mode, CudaPointer alpha, cudnnTensorDescriptor yDesc, CudaPointer y,
                                  cudnnTensorDescriptor dyDesc, CudaPointer dy, CudaPointer beta, cudnnTensorDescriptor dxDesc, CudaPointer dx) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSoftmaxBackward(this.handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc,
        dx);
    cudnnSoftmaxBackward_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSoftmaxBackward", result,
        new Object[]{this, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx});
    return result;
  }

  public int cudnnTransformTensor(final CudaPointer alpha, final cudnnTensorDescriptor xDesc, final CudaPointer x,
                                  final CudaPointer beta, final cudnnTensorDescriptor yDesc, final CudaPointer y) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnTransformTensor(this.handle, alpha, xDesc, x, beta, yDesc, y);
    cudnnTransformTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnTransformTensor", result, new Object[]{this, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }

  public int cudnnSetTensor(cudnnTensorDescriptor yDesc, CudaPointer y, CudaPointer valuePtr) {
    assert CudaDevice.isThreadDeviceId(getDeviceId());
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnSetTensor(this.handle, yDesc, y, valuePtr);
    cudnnSetTensor_execution.accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnSetTensor", result, new Object[]{this, yDesc, y, valuePtr});
    return result;
  }

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
          new Object[]{this, filterDesc, dyDesc, convDesc, dxDesc, algorithm, sizeInBytesArray});
      CudaSystem.handle(result);
      size = sizeInBytesArray[0];
    } catch (Throwable e) {
      logger.info("Error in allocateBackwardDataWorkspace", e);
      size = 0;
    }
    return this.allocate(Math.max(minSize, size), MemoryType.Device, true);
  }

  public void cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor reduceTensorDesc) {
    long startTime = RefSystem.nanoTime();
    final int result = JCudnn.cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    cudnnCreateReduceTensorDescriptor_execution
        .accept((RefSystem.nanoTime() - startTime) / 1e9);
    log("cudnnCreateReduceTensorDescriptor", result, new Object[]{reduceTensorDesc});
  }

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
        + Long.toHexString(RefSystem.identityHashCode(this));
  }

  @Nonnull
  public CudaResource<cudnnReduceTensorDescriptor> cudnnCreateReduceTensorDescriptor(int reduceTensorOp,
                                                                                     int reduceTensorCompType, int reduceTensorNanOpt, int reduceTensorIndices, int reduceTensorIndicesType) {
    cudnnReduceTensorDescriptor reduceTensorDesc = new cudnnReduceTensorDescriptor();
    cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType);
    return new CudaResource<cudnnReduceTensorDescriptor>(reduceTensorDesc,
        CudnnHandle::cudnnDestroyReduceTensorDescriptor, getDeviceId());
  }

  @Nonnull
  @Override
  public CudnnHandle addRef() {
    return (CudnnHandle) super.addRef();
  }

  @Override
  protected void cleanup() {
    RefArrayList<CudaResourceBase> objsToFree = new RefArrayList<>();
    cleanupNative.drainTo(objsToFree);
    if (objsToFree.isEmpty()) {
      objsToFree.freeRef();
      return;
    }

    if (CudaMemory.METRICS.get(deviceId).load() < CudaSettings.INSTANCE().asyncFreeLoadThreshold) {
      cleanupPool.submit(RefUtil.wrapInterface((Runnable) () -> {
        if (CudaSettings.INSTANCE().isSyncBeforeFree())
          synchronize(RefSystem.nanoTime(), deviceId);
        objsToFree.stream().forEach(CudaResourceBase::release);
        super.cleanup();
      }, objsToFree.addRef()));
    } else {
      if (CudaSettings.INSTANCE().isSyncBeforeFree())
        synchronize(RefSystem.nanoTime(), deviceId);
      objsToFree.stream().forEach(CudaResourceBase::release);
      super.cleanup();
    }
    objsToFree.freeRef();
  }

  @Override
  protected void _free() {
    cleanupNative.freeRef();
    final int result = JCudnn.cudnnDestroy(handle);
    log("cudnnDestroy", result, new Object[]{handle});
    CudaSystem.handle(result);
    super._free();
  }
}
