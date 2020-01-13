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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.function.Function;
import java.util.function.IntFunction;

public class CudaTensorList extends RegisteredObjectBase implements TensorList, CudaSystem.CudaDeviceResource {
  public static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace()
      : new StackTraceElement[] {};
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
      if (null != ptr)
        ptr.freeRef();
      throw new IllegalArgumentException("ptr");
    }
    if (null == ptr.memory.getPtr()) {
      if (null != ptr)
        ptr.freeRef();
      throw new IllegalArgumentException("ptr.getPtr()");
    }
    if (length <= 0) {
      if (null != ptr)
        ptr.freeRef();
      throw new IllegalArgumentException();
    }
    if (Tensor.length(dimensions) <= 0) {
      if (null != ptr)
        ptr.freeRef();
      throw new IllegalArgumentException();
    }
    CudaTensor temp_07_0001 = ptr == null ? null : ptr.addRef();
    if (null != this.gpuCopy)
      this.gpuCopy.freeRef();
    this.gpuCopy = temp_07_0001 == null ? null : temp_07_0001.addRef();
    if (null != temp_07_0001)
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
    if (null != ptr)
      ptr.freeRef();
    register();
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }

  public int getDeviceId() {
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    int temp_07_0007 = null == gpuCopy ? -1 : gpuCopy.memory.getDeviceId();
    if (null != gpuCopy)
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
    Precision temp_07_0008 = null == gpuCopy ? Precision.Double : gpuCopy.getPrecision();
    if (null != gpuCopy)
      gpuCopy.freeRef();
    return temp_07_0008;
  }

  public static long evictToHeap(int deviceId) {
    return CudaSystem.withDevice(deviceId, gpu -> {
      long size = RegisteredObjectBase.getLivingInstances(CudaTensorList.class).filter(x -> {
        boolean temp_07_0009 = x.gpuCopy != null
            && (x.getDeviceId() == deviceId || deviceId < 0 || x.getDeviceId() < 0);
        if (null != x)
          x.freeRef();
        return temp_07_0009;
      }).mapToLong(CudaTensorList::evictToHeap).sum();
      logger.debug(RefString.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
      return size;
    });
  }

  public static CudaTensorList create(final CudaMemory ptr, CudaDevice.CudaTensorDescriptor descriptor,
      final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    CudaTensor cudaTensor = new CudaTensor(ptr == null ? null : ptr.addRef(),
        descriptor == null ? null : descriptor.addRef(), precision);
    if (null != descriptor)
      descriptor.freeRef();
    if (null != ptr)
      ptr.freeRef();
    CudaTensorList temp_07_0010 = new CudaTensorList(cudaTensor == null ? null : cudaTensor.addRef(), length,
        dimensions, precision);
    if (null != cudaTensor)
      cudaTensor.freeRef();
    return temp_07_0010;
  }

  public static @SuppressWarnings("unused") CudaTensorList[] addRefs(CudaTensorList[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaTensorList::addRef)
        .toArray((x) -> new CudaTensorList[x]);
  }

  public static @SuppressWarnings("unused") CudaTensorList[][] addRefs(CudaTensorList[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaTensorList::addRefs)
        .toArray((x) -> new CudaTensorList[x][]);
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
        return add(right == null ? null : right.addRef());
      }
      assert length() == right.length();
      if (heapCopy == null) {
        if (right instanceof CudaTensorList) {
          @Nonnull
          final CudaTensorList nativeRight = (CudaTensorList) right.addRef();
          if (nativeRight.getPrecision() == this.getPrecision()) {
            if (nativeRight.heapCopy == null) {
              assert (nativeRight.gpuCopy != this.gpuCopy);
              if (null != this.gpuCopy.memory) {
                int deviceId = this.gpuCopy.memory.getDeviceId();
                TensorList temp_07_0011 = CudaSystem
                    .run(RefUtil.wrapInterface((Function<CudnnHandle, TensorList>) gpu -> {
                      if (gpu.getDeviceId() == deviceId) {
                        return gpu.addInPlace(this.addRef(), nativeRight == null ? null : nativeRight.addRef());
                      } else {
                        assertAlive();
                        right.assertAlive();
                        return add(right == null ? null : right.addRef());
                      }
                    }, nativeRight == null ? null : nativeRight, right == null ? null : right.addRef()), this.addRef(),
                        right == null ? null : right.addRef());
                return temp_07_0011;
              }
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
            Tensor a = get(i);
            Tensor b = right.get(i);
            Tensor temp_07_0012 = a.addAndFree(b == null ? null : b.addRef());
            if (null != b)
              b.freeRef();
            if (null != a)
              a.freeRef();
            return temp_07_0012;
          }, right == null ? null : right.addRef())).toArray(i -> new Tensor[i]));
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
          @Nonnull
          final CudaTensorList nativeRight = (CudaTensorList) right.addRef();
          if (nativeRight.getPrecision() == this.getPrecision()) {
            if (nativeRight.heapCopy == null) {
              CudaTensorList temp_07_0013 = CudaSystem
                  .run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensorList>) gpu -> {
                    return gpu.add(this.addRef(), nativeRight == null ? null : nativeRight.addRef());
                  }, nativeRight == null ? null : nativeRight), this.addRef());
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
            Tensor a = get(i);
            Tensor b = right.get(i);
            Tensor temp_07_0014 = a.addAndFree(b == null ? null : b.addRef());
            if (null != b)
              b.freeRef();
            if (null != a)
              a.freeRef();
            return temp_07_0014;
          }, right == null ? null : right.addRef())).toArray(i -> new Tensor[i]));
      return temp_07_0020;
    } finally {
      right.freeRef();
    }
  }

  @Override
  @Nonnull
  public Tensor get(final int i) {
    assertAlive();
    if (heapCopy != null)
      return heapCopy.get(i);
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    Tensor temp_07_0015 = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, Tensor>) gpu -> {
      TimedResult<Tensor> timedResult = TimedResult.time(RefUtil.wrapInterface((UncheckedSupplier<Tensor>) () -> {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        Tensor t = new Tensor(getDimensions());
        if (gpuCopy.isDense()) {
          CudaMemory memory = gpuCopy.getMemory(gpu);
          RefUtil.freeRef(memory.read(gpuCopy.getPrecision(), t.getData(), i * Tensor.length(getDimensions())));
          if (null != memory)
            memory.freeRef();
        } else {
          gpuCopy.read(gpu, i, t == null ? null : t.addRef(), false);
        }
        return t;
      }, gpuCopy == null ? null : gpuCopy.addRef()));
      CudaTensorList.logger.debug(RefString.format("Read %s bytes in %.4f from Tensor %s, GPU at %s, created by %s",
          gpuCopy.size(), timedResult.seconds(),
          Integer.toHexString(com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(timedResult.result)),
          Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
          Util.toString(createdBy).replaceAll("\n", "\n\t")));
      return timedResult.result;
    }, gpuCopy == null ? null : gpuCopy.addRef()), CudaTensorList.this.addRef());
    if (null != gpuCopy)
      gpuCopy.freeRef();
    return temp_07_0015;
  }

  @Override
  public int length() {
    return getLength();
  }

  @Override
  public RefStream<Tensor> stream() {
    TensorArray temp_07_0023 = heapCopy();
    RefStream<Tensor> temp_07_0022 = temp_07_0023.stream();
    if (null != temp_07_0023)
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
      CudaMemory copyPtr = cudaMemory.copy(gpu, MemoryType.Managed.ifEnabled());
      if (null != cudaMemory)
        cudaMemory.freeRef();
      CudaTensor cudaTensor = new CudaTensor(copyPtr == null ? null : copyPtr.addRef(), ptr.descriptor.addRef(),
          getPrecision());
      if (null != ptr)
        ptr.freeRef();
      if (null != copyPtr)
        copyPtr.freeRef();
      CudaTensorList temp_07_0016 = new CudaTensorList(cudaTensor == null ? null : cudaTensor.addRef(), getLength(),
          getDimensions(), getPrecision());
      if (null != cudaTensor)
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
    if (null != temp_07_0024)
      temp_07_0024.freeRef();
    CudaTensor ptr;
    synchronized (this) {
      ptr = this.gpuCopy.addRef();
      CudaTensor temp_07_0002 = null;
      if (null != this.gpuCopy)
        this.gpuCopy.freeRef();
      this.gpuCopy = temp_07_0002 == null ? null : temp_07_0002.addRef();
      if (null != temp_07_0002)
        temp_07_0002.freeRef();
    }
    if (null != ptr && !ptr.isFinalized()) {
      long elements = getElements();
      assert 0 < length;
      assert 0 < elements : RefArrays.toString(dimensions);
      if (null != ptr)
        ptr.freeRef();
      return elements * getPrecision().size;
    } else {
      synchronized (this) {
        CudaTensor temp_07_0003 = ptr == null ? null : ptr.addRef();
        if (null != this.gpuCopy)
          this.gpuCopy.freeRef();
        this.gpuCopy = temp_07_0003 == null ? null : temp_07_0003.addRef();
        if (null != temp_07_0003)
          temp_07_0003.freeRef();
      }
      if (null != ptr)
        ptr.freeRef();
      return 0;
    }
  }

  public void _free() {
    if (null != heapCopy)
      heapCopy.freeRef();
    heapCopy = null;
    if (null != gpuCopy)
      gpuCopy.freeRef();
    gpuCopy = null;
    synchronized (this) {
      if (null != gpuCopy) {
        CudaTensor temp_07_0004 = null;
        if (null != gpuCopy)
          gpuCopy.freeRef();
        gpuCopy = temp_07_0004 == null ? null : temp_07_0004.addRef();
        if (null != temp_07_0004)
          temp_07_0004.freeRef();
      }
      if (null != heapCopy) {
        TensorArray temp_07_0005 = null;
        if (null != heapCopy)
          heapCopy.freeRef();
        heapCopy = temp_07_0005 == null ? null : temp_07_0005.addRef();
        if (null != temp_07_0005)
          temp_07_0005.freeRef();
      }
    }
  }

  public @Override @SuppressWarnings("unused") CudaTensorList addRef() {
    return (CudaTensorList) super.addRef();
  }

  @Nullable
  private TensorArray heapCopy() {
    return heapCopy(false);
  }

  @Nullable
  private TensorArray heapCopy(final boolean avoidAllocations) {
    TensorArray heapCopy = this.heapCopy.addRef();
    if (null == heapCopy || heapCopy.isFinalized()) {
      assertAlive();
      TensorArray copy = toHeap(avoidAllocations);
      final TensorArray prev;
      synchronized (this) {
        heapCopy = this.heapCopy.addRef();
        if (copy == heapCopy) {
          prev = null;
        } else if (null == heapCopy || heapCopy.isFinalized()) {
          prev = this.heapCopy.addRef();
          TensorArray temp_07_0006 = copy == null ? null : copy.addRef();
          if (null != this.heapCopy)
            this.heapCopy.freeRef();
          this.heapCopy = temp_07_0006 == null ? null : temp_07_0006.addRef();
          if (null != temp_07_0006)
            temp_07_0006.freeRef();
          heapCopy = copy == null ? null : copy.addRef();
        } else {
          prev = null;
        }
      }
      if (null != prev)
        prev.freeRef();
      if (null != copy)
        copy.freeRef();
    }
    return heapCopy;
  }

  private TensorArray toHeap(final boolean avoidAllocations) {
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    if (null == gpuCopy || !gpuCopy.tryAddRef()) {
      if (null == heapCopy) {
        if (null != gpuCopy)
          gpuCopy.freeRef();
        throw new IllegalStateException("No data");
      } else if (heapCopy.isFinalized()) {
        throw new IllegalStateException("Local data has been freed");
      } else {
        return heapCopy == null ? null : heapCopy.addRef();
      }
    }
    int length = getLength();
    if (0 >= length) {
      if (null != gpuCopy)
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
              return new TensorArray(Tensor.addRefs(output));
            }, Tensor.addRefs(output), gpuCopy == null ? null : gpuCopy.addRef()), this.addRef()),
            Tensor.addRefs(output), gpuCopy == null ? null : gpuCopy.addRef()));
    if (null != output)
      ReferenceCounting.freeRefs(output);
    CudaTensorList.logger.debug(RefString.format("Read %s bytes in %.4f from Tensor %s on GPU at %s, created by %s",
        gpuCopy.size(), timedResult.seconds(),
        Integer.toHexString(com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(timedResult.result)),
        Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
        Util.toString(createdBy).replaceAll("\n", "\n\t")));
    if (null != gpuCopy)
      gpuCopy.freeRef();
    return timedResult.result;
  }
}
