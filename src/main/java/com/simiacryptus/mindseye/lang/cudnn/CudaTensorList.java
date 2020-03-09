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
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.IntFunction;

public class CudaTensorList extends ReferenceCountingBase implements TensorList, CudaSystem.CudaDeviceResource {
  public static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace()
      : new StackTraceElement[]{};
  @Nonnull
  private final int[] dimensions;
  private final int length;
  @Nullable
  volatile CudaTensor cudaTensor;
  @Nullable
  volatile TensorArray heapCopy = null;

  public CudaTensorList(@Nullable final CudaTensor cudaTensor, final int length, @Nonnull final int[] dimensions,
                        @Nonnull final Precision precision) {
    //assert 1 == ptr.currentRefCount() : ptr.referenceReport(false, false);
    if (null == cudaTensor) {
      throw new IllegalArgumentException("ptr");
    }
    if (null == cudaTensor.memory.getPtr()) {
      cudaTensor.freeRef();
      throw new IllegalArgumentException("ptr.getPtr()");
    }
    if (length <= 0) {
      cudaTensor.freeRef();
      throw new IllegalArgumentException();
    }
    if (Tensor.length(dimensions) <= 0) {
      cudaTensor.freeRef();
      throw new IllegalArgumentException();
    }
    if (cudaTensor.memory.size < (long) (length - 1) * Tensor.length(dimensions) * precision.size) {
      String message = String.format("%s < %s", cudaTensor.memory.size, (long) length * Tensor.length(dimensions) * precision.size);
      cudaTensor.freeRef();
      throw new AssertionError(message);
    }
    if (cudaTensor.descriptor.batchCount != length) {
      cudaTensor.freeRef();
      throw new AssertionError();
    }
    if (cudaTensor.descriptor.channels != (dimensions.length < 3 ? 1 : dimensions[2])) {
      String message = RefString.format(
          "%s != (%d,%d,%d,%d)", RefArrays.toString(dimensions), cudaTensor.descriptor.batchCount, cudaTensor.descriptor.channels,
          cudaTensor.descriptor.height, cudaTensor.descriptor.width);
      cudaTensor.freeRef();
      throw new AssertionError(message);
    }
    if (cudaTensor.descriptor.height != (dimensions.length < 2 ? 1 : dimensions[1])) {
      String message = RefString.format(
          "%s != (%d,%d,%d,%d)", RefArrays.toString(dimensions), cudaTensor.descriptor.batchCount, cudaTensor.descriptor.channels,
          cudaTensor.descriptor.height, cudaTensor.descriptor.width);
      cudaTensor.freeRef();
      throw new AssertionError(message);
    }
    if (cudaTensor.descriptor.width != (dimensions.length < 1 ? 1 : dimensions[0])) {
      String message = RefString.format("%s != (%d,%d,%d,%d)",
          RefArrays.toString(dimensions), cudaTensor.descriptor.batchCount, cudaTensor.descriptor.channels, cudaTensor.descriptor.height,
          cudaTensor.descriptor.width);
      cudaTensor.freeRef();
      throw new AssertionError(message);
    }
    if (cudaTensor.getPrecision() != precision) {
      cudaTensor.freeRef();
      throw new AssertionError();
    }
    if (cudaTensor.memory.getPtr() == null) {
      cudaTensor.freeRef();
      throw new AssertionError();
    }
    setCudaTensor(cudaTensor);
    this.dimensions = RefArrays.copyOf(dimensions, dimensions.length);
    this.length = length;
    ObjectRegistry.register(this.addRef());
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }

  public int getDeviceId() {
    return null == cudaTensor ? -1 : cudaTensor.getDeviceId();
  }

  @Nonnull
  @Override
  public int[] getDimensions() {
    return RefArrays.copyOf(dimensions, dimensions.length);
  }

  public int getLength() {
    return length;
  }

  /**
   * The Precision.
   */
  @Nonnull
  public Precision getPrecision() {
    return null == cudaTensor ? null : cudaTensor.getPrecision();
  }

  void setCudaTensor(CudaTensor cudaTensor) {
    synchronized (this) {
      if (cudaTensor != this.cudaTensor) {
        RefUtil.freeRef(this.cudaTensor);
        this.cudaTensor = cudaTensor;
      } else {
        cudaTensor.freeRef();
      }
    }
  }

  private synchronized void setHeapCopy(TensorArray toHeap) {
    if (toHeap != heapCopy && !hasHeapCopy()) {
      if (hasHeapCopy()) {
        this.heapCopy.freeRef();
      }
      this.heapCopy = toHeap;
    } else if (null != toHeap) {
      toHeap.freeRef();
    }
  }

  public static long evictToHeap(int deviceId) {
    return CudaSystem.withDevice(deviceId, gpu -> {
      long size = ObjectRegistry.getLivingInstances(CudaTensorList.class).filter(cudaTensorList -> {
        int tensorDevice = cudaTensorList.getDeviceId();
        boolean inDevice = cudaTensorList.cudaTensor != null
            && (tensorDevice == deviceId || deviceId < 0 || tensorDevice < 0);
        cudaTensorList.freeRef();
        return inDevice;
      }).mapToLong(cudaTensorList -> {
        long toHeap = cudaTensorList.evictToHeap();
        cudaTensorList.freeRef();
        return toHeap;
      }).sum();
      gpu.freeRef();
      logger.debug(RefString.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
      return size;
    });
  }

  @Nonnull
  public static CudaTensorList create(@Nullable final CudaMemory ptr, @Nullable CudaDevice.CudaTensorDescriptor descriptor,
                                      final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    return new CudaTensorList(
        new CudaTensor(ptr, descriptor, precision),
        length, dimensions, precision);
  }

  @Override
  public TensorList addAndFree(@Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    try {
      if (right instanceof ReshapedTensorList) {
        return addAndFree(((ReshapedTensorList) right).getInner());
      }
      if (1 < currentRefCount()) {
        return add(right.addRef());
      }
      assert length() == right.length();
      if (heapCopy == null) {
        if (right instanceof CudaTensorList) {
          @Nonnull final CudaTensorList nativeRight = (CudaTensorList) right.addRef();
          if (nativeRight.getPrecision() == this.getPrecision()) {
            if (nativeRight.heapCopy == null) {
              assert nativeRight.cudaTensor != this.cudaTensor;
              int deviceId = this.cudaTensor.getDeviceId();
              return CudaSystem
                  .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, TensorList>) gpu -> {
                        try {
                          if (gpu.getDeviceId() == deviceId) {
                            return gpu.addInPlace(this.addRef(), nativeRight.addRef());
                          } else {
                            assertAlive();
                            right.assertAlive();
                            return add(right.addRef());
                          }
                        } finally {
                          gpu.freeRef();
                        }
                      }, nativeRight, right.addRef()), this.addRef(),
                      right.addRef());
            }
          }
          nativeRight.freeRef();
        }
      }
      if (right.length() == 0) {
        return this.addRef();
      }
      if (length() == 0) {
        throw new IllegalArgumentException();
      }
      assert length() == right.length();
      return new TensorArray(
          RefIntStream.range(0, length()).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            return Tensor.add(get(i), right.get(i));
          }, right.addRef())).toArray(i -> new Tensor[i]));
    } finally {
      right.freeRef();
    }
  }

  @Override
  public TensorList add(@Nonnull final TensorList right) {
    try {
      assertAlive();
      right.assertAlive();
      assert length() == right.length();
      if (right instanceof ReshapedTensorList) {
        return add(((ReshapedTensorList) right).getInner());
      }
      if (heapCopy == null) {
        if (right instanceof CudaTensorList) {
          @Nonnull final CudaTensorList nativeRight = (CudaTensorList) right.addRef();
          if (nativeRight.getPrecision() == this.getPrecision()) {
            if (nativeRight.heapCopy == null) {
              return CudaSystem
                  .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                    CudaTensorList add = gpu.add(this.addRef(), nativeRight.addRef());
                    gpu.freeRef();
                    return add;
                  }, nativeRight), this.addRef());
            }
          }
          nativeRight.freeRef();
        }
      }
      if (right.length() == 0) {
        return this.addRef();
      }
      if (length() == 0) {
        throw new IllegalArgumentException();
      }
      assert length() == right.length();
      return new TensorArray(
          RefIntStream.range(0, length()).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            return Tensor.add(get(i), right.get(i));
          }, right.addRef())).toArray(i -> new Tensor[i]));
    } finally {
      right.freeRef();
    }
  }

  @Override
  @Nonnull
  @RefAware
  public Tensor get(final int i) {
    assertAlive();
    if (heapCopy != null)
      return heapCopy.get(i);
    CudaTensor cudaTensor = this.cudaTensor.addRef();
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, Tensor>) gpu -> {
      TimedResult<Tensor> timedResult = TimedResult.time(RefUtil.wrapInterface((UncheckedSupplier<Tensor>) () -> {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        Tensor t = new Tensor(getDimensions());
        if (cudaTensor.isDense()) {
          CudaMemory memory = cudaTensor.getMemory(gpu.addRef());
          assert memory != null;
          memory.read(cudaTensor.getPrecision(), t.addRef(), i * Tensor.length(getDimensions()));
          memory.freeRef();
        } else {
          cudaTensor.read(gpu.addRef(), i, t.addRef(), false);
        }
        return t;
      }, cudaTensor.addRef(), gpu));
      Tensor result = timedResult.getResult();
      if (CudaTensorList.logger.isDebugEnabled()) CudaTensorList.logger.debug(RefString.format(
          "Read %s bytes in %.4f from Tensor %s, GPU at %s, created by %s",
          cudaTensor.size(),
          timedResult.seconds(),
          Integer.toHexString(RefSystem.identityHashCode(RefUtil.addRef(result))),
          Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
          Util.toString(createdBy).replaceAll("\n", "\n\t")
      ));
      timedResult.freeRef();
      return result;
    }, cudaTensor), this.addRef());
  }

  @Override
  public int length() {
    return getLength();
  }

  @Nonnull
  @Override
  public RefStream<Tensor> stream() {
    TensorArray heapCopy = heapCopy();
    assert heapCopy != null;
    RefStream<Tensor> stream = heapCopy.stream();
    heapCopy.freeRef();
    return stream;
  }

  @Override
  public TensorList copy() {
    return CudaSystem.run(gpu -> {
      CudaTensor ptr = gpu.getTensor(this.addRef(), MemoryType.Device, false);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      CudaMemory cudaMemory = ptr.getMemory(gpu.addRef(), MemoryType.Device);
      assert cudaMemory != null;
      CudaMemory copyPtr = cudaMemory.copy(gpu, MemoryType.Managed.ifEnabled());
      cudaMemory.freeRef();
      Precision precision = getPrecision();
      CudaTensor cudaTensor = new CudaTensor(copyPtr, ptr.descriptor.addRef(), precision);
      ptr.freeRef();
      return new CudaTensorList(cudaTensor, getLength(), getDimensions(), precision);
    }, this.addRef());
  }

  @RefIgnore
  public long evictToHeap() {
    if (isFreed())
      return 0;
    TensorArray heapCopy = heapCopy(true);
    if (null == heapCopy) {
      throw new IllegalStateException();
    }
    heapCopy.freeRef();
    final CudaTensor ptr;
    synchronized (this) {
      ptr = this.cudaTensor;
      this.cudaTensor = null;
    }
    if (null != ptr && !ptr.isFreed()) {
      long elements = getElements();
      assert 0 < length;
      assert 0 < elements : RefArrays.toString(dimensions);
      long size = elements * ptr.getPrecision().size;
      ptr.freeRef();
      return size;
    } else {
      synchronized (this) {
        if (null != this.cudaTensor)
          this.cudaTensor.freeRef();
        this.cudaTensor = ptr;
      }
      return 0;
    }
  }

  public void _free() {
    super._free();
    synchronized (this) {
      if (null != cudaTensor) {
        cudaTensor.freeRef();
        cudaTensor = null;
      }
      if (null != heapCopy) {
        heapCopy.freeRef();
        heapCopy = null;
      }
    }
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CudaTensorList addRef() {
    return (CudaTensorList) super.addRef();
  }

  @Nullable
  private TensorArray heapCopy() {
    return heapCopy(false);
  }

  @Nullable
  private TensorArray heapCopy(final boolean avoidAllocations) {
    if (!hasHeapCopy()) {
      setHeapCopy(toHeap(avoidAllocations));
    }
    return this.heapCopy.addRef();
  }

  private boolean hasHeapCopy() {
    return null != heapCopy && !heapCopy.isFreed();
  }

  @Nullable
  @RefIgnore
  private TensorArray toHeap(final boolean avoidAllocations) {
    assertAlive();
    CudaTensor cudaTensor = this.cudaTensor;
    if (cudaTensor.tryAddRef()) {
      return toHeap(avoidAllocations, cudaTensor);
    } else {
      if (null == heapCopy) {
        throw new IllegalStateException("No data");
      } else if (heapCopy.isFreed()) {
        throw new IllegalStateException("Local data has been freed");
      } else {
        return heapCopy.addRef();
      }
    }
  }

  private TensorArray toHeap(boolean avoidAllocations, CudaTensor gpuCopy) {
    try {
      if (null == gpuCopy) return null;
      int length = getLength();
      if (0 >= length) {
        throw new IllegalStateException();
      }
      final Tensor[] output = RefIntStream.range(0, length)
          .mapToObj(dataIndex -> new Tensor(getDimensions()))
          .toArray(i -> new Tensor[i]);
      TimedResult<TensorArray> timedResult = TimedResult
          .time(RefUtil.wrapInterface((UncheckedSupplier<TensorArray>) () -> CudaDevice
              .run(gpu -> {
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    //        assert getPrecision() == gpuCopy.getPrecision();
                    //        assert getPrecision() == gpuCopy.descriptor.dataType;
                    for (int i = 0; i < length; i++) {
                      gpuCopy.read(gpu.addRef(), i, output[i].addRef(), avoidAllocations);
                    }
                    gpu.freeRef();
                    return new TensorArray(RefUtil.addRef(output));
                  }, this.addRef()
              ), output));
      TensorArray result = timedResult.getResult();
      if (CudaTensorList.logger.isDebugEnabled()) {
        CudaTensorList.logger.debug(RefString.format(
            "Read %s bytes in %.4f from Tensor %s on GPU at %s, created by %s",
            gpuCopy.size(),
            timedResult.seconds(),
            Integer.toHexString(RefSystem.identityHashCode(result.addRef())),
            Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
            Util.toString(createdBy).replaceAll("\n", "\n\t"))
        );
      }
      timedResult.freeRef();
      return result;
    } finally {
      if (null != gpuCopy) gpuCopy.freeRef();
    }
  }
}
