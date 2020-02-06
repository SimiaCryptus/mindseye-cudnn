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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.Function;
import java.util.function.IntFunction;

public class CudaTensorList extends ReferenceCountingBase implements TensorList, CudaSystem.CudaDeviceResource {
  public static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace()
      : new StackTraceElement[]{};
  @Nonnull
  private final int[] dimensions;
  private final int length;
  @Nullable
  volatile CudaTensor gpuCopy;
  @Nullable
  volatile TensorArray heapCopy = null;

  public CudaTensorList(@Nullable final CudaTensor ptr, final int length, @Nonnull final int[] dimensions,
                        @Nonnull final Precision precision) {
    //assert 1 == ptr.currentRefCount() : ptr.referenceReport(false, false);
    if (null == ptr) {
      throw new IllegalArgumentException("ptr");
    }
    if (null == ptr.memory.getPtr()) {
      ptr.freeRef();
      throw new IllegalArgumentException("ptr.getPtr()");
    }
    if (length <= 0) {
      ptr.freeRef();
      throw new IllegalArgumentException();
    }
    if (Tensor.length(dimensions) <= 0) {
      ptr.freeRef();
      throw new IllegalArgumentException();
    }
    CudaTensor temp_07_0001 = ptr.addRef();
    if (null != this.gpuCopy)
      this.gpuCopy.freeRef();
    this.gpuCopy = temp_07_0001.addRef();
    temp_07_0001.freeRef();
    this.dimensions = RefArrays.copyOf(dimensions, dimensions.length);
    this.length = length;
    assert ptr.memory.size >= (long) (length - 1) * Tensor.length(dimensions) * precision.size : String
        .format("%s < %s", ptr.memory.size, (long) length * Tensor.length(dimensions) * precision.size);
    assert ptr.descriptor.batchCount == length;
    assert ptr.descriptor.channels == (dimensions.length < 3 ? 1 : dimensions[2]) : RefString.format(
        "%s != (%d,%d,%d,%d)", RefArrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels,
        ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.height == (dimensions.length < 2 ? 1 : dimensions[1]) : RefString.format(
        "%s != (%d,%d,%d,%d)", RefArrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels,
        ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.width == (dimensions.length < 1 ? 1 : dimensions[0]) : RefString.format("%s != (%d,%d,%d,%d)",
        RefArrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height,
        ptr.descriptor.width);
    assert ptr.getPrecision() == precision;
    assert ptr.memory.getPtr() != null;
    ptr.freeRef();
    ObjectRegistry.register(this);
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }

  public int getDeviceId() {
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    int temp_07_0007 = gpuCopy.memory.getDeviceId();
    gpuCopy.freeRef();
    return temp_07_0007;
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
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    Precision temp_07_0008 = gpuCopy.getPrecision();
    gpuCopy.freeRef();
    return temp_07_0008;
  }

  public static long evictToHeap(int deviceId) {
    return CudaSystem.withDevice(deviceId, gpu -> {
      long size = ObjectRegistry.getLivingInstances(CudaTensorList.class).filter(x -> {
        boolean temp_07_0009 = x.gpuCopy != null
            && (x.getDeviceId() == deviceId || deviceId < 0 || x.getDeviceId() < 0);
        x.freeRef();
        return temp_07_0009;
      }).mapToLong(cudaTensorList -> cudaTensorList.evictToHeap()).sum();
      logger.debug(RefString.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
      return size;
    });
  }

  @Nonnull
  public static CudaTensorList create(@Nullable final CudaMemory ptr, @Nullable CudaDevice.CudaTensorDescriptor descriptor,
                                      final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    CudaTensor cudaTensor = new CudaTensor(ptr == null ? null : ptr.addRef(),
        descriptor == null ? null : descriptor.addRef(), precision);
    if (null != descriptor)
      descriptor.freeRef();
    if (null != ptr)
      ptr.freeRef();
    CudaTensorList temp_07_0010 = new CudaTensorList(cudaTensor.addRef(), length,
        dimensions, precision);
    cudaTensor.freeRef();
    return temp_07_0010;
  }

  @Override
  public TensorList addAndFree(@Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    try {
      if (right instanceof ReshapedTensorList) {
        TensorList temp_07_0018 = addAndFree(((ReshapedTensorList) right).getInner());
        return temp_07_0018;
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
              assert nativeRight.gpuCopy != this.gpuCopy;
              int deviceId = this.gpuCopy.memory.getDeviceId();
              TensorList temp_07_0011 = CudaSystem
                  .run(RefUtil.wrapInterface((Function<CudnnHandle, TensorList>) gpu -> {
                        if (gpu.getDeviceId() == deviceId) {
                          return gpu.addInPlace(this.addRef(), nativeRight.addRef());
                        } else {
                          assertAlive();
                          right.assertAlive();
                          return add(right.addRef());
                        }
                      }, nativeRight, right.addRef()), this.addRef(),
                      right.addRef());
              return temp_07_0011;
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
      TensorArray temp_07_0017 = new TensorArray(
          RefIntStream.range(0, length()).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            return Tensor.add(get(i), right.get(i));
          }, right.addRef())).toArray(i -> new Tensor[i]));
      return temp_07_0017;
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
              CudaTensorList temp_07_0013 = CudaSystem
                  .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                    return gpu.add(this.addRef(), nativeRight.addRef());
                  }, nativeRight), this.addRef());
              return temp_07_0013;
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
      TensorArray temp_07_0020 = new TensorArray(
          RefIntStream.range(0, length()).mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            return Tensor.add(get(i), right.get(i));
          }, right.addRef())).toArray(i -> new Tensor[i]));
      return temp_07_0020;
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
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    return CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, Tensor>) gpu -> {
      TimedResult<Tensor> timedResult = TimedResult.time(RefUtil.wrapInterface((UncheckedSupplier<Tensor>) () -> {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        Tensor t = new Tensor(getDimensions());
        if (gpuCopy.isDense()) {
          CudaMemory memory = gpuCopy.getMemory(gpu.addRef());
          assert memory != null;
          memory.read(gpuCopy.getPrecision(), t.getData(), i * Tensor.length(getDimensions()));
          memory.freeRef();
        } else {
          gpuCopy.read(gpu.addRef(), i, t.addRef(), false);
        }
        return t;
      }, gpuCopy.addRef(), gpu));
      if(CudaTensorList.logger.isDebugEnabled()) CudaTensorList.logger.debug(RefString.format("Read %s bytes in %.4f from Tensor %s, GPU at %s, created by %s",
          gpuCopy.size(), timedResult.seconds(),
          Integer.toHexString(RefSystem.identityHashCode(timedResult.getResult())),
          Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
          Util.toString(createdBy).replaceAll("\n", "\n\t")));
      Tensor result = timedResult.getResult();
      timedResult.freeRef();
      return result;
    }, gpuCopy), this.addRef());
  }

  @Override
  public int length() {
    return getLength();
  }

  @Nonnull
  @Override
  public RefStream<Tensor> stream() {
    TensorArray temp_07_0023 = heapCopy();
    assert temp_07_0023 != null;
    RefStream<Tensor> temp_07_0022 = temp_07_0023.stream();
    temp_07_0023.freeRef();
    return temp_07_0022;
  }

  @Override
  public TensorList copy() {
    return CudaSystem.run(gpu -> {
      CudaTensor ptr = gpu.getTensor(this.addRef(), MemoryType.Device, false);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      CudaMemory cudaMemory = ptr.getMemory(gpu, MemoryType.Device);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      assert cudaMemory != null;
      CudaMemory copyPtr = cudaMemory.copy(gpu, MemoryType.Managed.ifEnabled());
      cudaMemory.freeRef();
      CudaTensor cudaTensor = new CudaTensor(copyPtr.addRef(), ptr.descriptor.addRef(),
          getPrecision());
      ptr.freeRef();
      copyPtr.freeRef();
      CudaTensorList temp_07_0016 = new CudaTensorList(cudaTensor.addRef(), getLength(),
          getDimensions(), getPrecision());
      cudaTensor.freeRef();
      return temp_07_0016;
    }, this.addRef());
  }

  public long evictToHeap() {
    if (isFinalized())
      return 0;
    TensorArray temp_07_0024 = heapCopy(true);
    if (null == temp_07_0024) {
      throw new IllegalStateException();
    }
    temp_07_0024.freeRef();
    CudaTensor ptr = null;
    synchronized (this) {
      RefUtil.freeRef(ptr);
      ptr = this.gpuCopy.addRef();
      if (null != this.gpuCopy)
        this.gpuCopy.freeRef();
      this.gpuCopy = null;
    }
    if (!ptr.isFinalized()) {
      long elements = getElements();
      assert 0 < length;
      assert 0 < elements : RefArrays.toString(dimensions);
      ptr.freeRef();
      return elements * getPrecision().size;
    } else {
      synchronized (this) {
        CudaTensor temp_07_0003 = ptr.addRef();
        if (null != this.gpuCopy)
          this.gpuCopy.freeRef();
        this.gpuCopy = temp_07_0003.addRef();
        temp_07_0003.freeRef();
      }
      ptr.freeRef();
      return 0;
    }
  }

  public void _free() {
    super._free();
    synchronized (this) {
      if (null != gpuCopy) {
        if (null != gpuCopy) {
          gpuCopy.freeRef();
          gpuCopy = null;
        }
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
    if (heapCopy.isFinalized()) {
      assertAlive();
      TensorArray copy = toHeap(avoidAllocations);
      synchronized (this) {
        if (copy != heapCopy && heapCopy.isFinalized()) {
          if (null != this.heapCopy)
            this.heapCopy.freeRef();
          this.heapCopy = copy;
        } else {
          if (null != copy)
            copy.freeRef();
        }
      }
    }
    return this.heapCopy.addRef();
  }

  @Nullable
  private TensorArray toHeap(final boolean avoidAllocations) {
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    if (!gpuCopy.tryAddRef()) {
      gpuCopy.freeRef();
      if (null == heapCopy) {
        throw new IllegalStateException("No data");
      } else if (heapCopy.isFinalized()) {
        throw new IllegalStateException("Local data has been freed");
      } else {
        return heapCopy.addRef();
      }
    }
    int length = getLength();
    if (0 >= length) {
      gpuCopy.freeRef();
      throw new IllegalStateException();
    }
    final Tensor[] output = RefIntStream.range(0, length).mapToObj(dataIndex -> new Tensor(getDimensions()))
        .toArray(i -> new Tensor[i]);
    TimedResult<TensorArray> timedResult = TimedResult
        .time(RefUtil.wrapInterface((UncheckedSupplier<TensorArray>) () -> CudaDevice
                .run(RefUtil.wrapInterface((Function<CudnnHandle, TensorArray>) gpu -> {
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  //        assert getPrecision() == gpuCopy.getPrecision();
                  //        assert getPrecision() == gpuCopy.descriptor.dataType;
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  for (int i = 0; i < length; i++) {
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    gpuCopy.read(gpu, i, output[i].addRef(), avoidAllocations);
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  }
                  return new TensorArray(RefUtil.addRefs(output));
                }, RefUtil.addRefs(output), gpuCopy.addRef()), this.addRef()),
            RefUtil.addRefs(output), gpuCopy.addRef()));
    RefUtil.freeRef(output);
    TensorArray result = timedResult.getResult();
    if(CudaTensorList.logger.isDebugEnabled()) CudaTensorList.logger.debug(RefString.format("Read %s bytes in %.4f from Tensor %s on GPU at %s, created by %s",
        gpuCopy.size(), timedResult.seconds(),
        Integer.toHexString(RefSystem.identityHashCode(result.addRef())),
        Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
        Util.toString(createdBy).replaceAll("\n", "\n\t")));
    timedResult.freeRef();
    gpuCopy.freeRef();
    return result;
  }
}
