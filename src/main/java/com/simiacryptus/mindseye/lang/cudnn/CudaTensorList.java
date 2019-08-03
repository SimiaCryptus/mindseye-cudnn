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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CudaTensorList extends RegisteredObjectBase implements TensorList, CudaSystem.CudaDeviceResource {
  public static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace() : new StackTraceElement[]{};
  @Nonnull
  private final int[] dimensions;
  private final int length;
  @Nullable
  volatile CudaTensor gpuCopy;
  @Nullable
  volatile TensorArray heapCopy = null;

  private CudaTensorList(@Nullable final CudaTensor ptr, final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    //assert 1 == ptr.currentRefCount() : ptr.referenceReport(false, false);
    if (null == ptr) throw new IllegalArgumentException("ptr");
    if (null == ptr.memory.getPtr()) throw new IllegalArgumentException("ptr.getPtr()");
    if (length <= 0) throw new IllegalArgumentException();
    if (Tensor.length(dimensions) <= 0) throw new IllegalArgumentException();
    this.gpuCopy = ptr;
    this.gpuCopy.addRef();
    this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
    this.length = length;
    assert ptr.memory.size >= (long) (length - 1) * Tensor.length(dimensions) * precision.size : String.format("%s < %s", ptr.memory.size, (long) length * Tensor.length(dimensions) * precision.size);
    assert ptr.descriptor.batchCount == length;
    assert ptr.descriptor.channels == (dimensions.length < 3 ? 1 : dimensions[2]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.height == (dimensions.length < 2 ? 1 : dimensions[1]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.width == (dimensions.length < 1 ? 1 : dimensions[0]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.getPrecision() == precision;
    assert ptr.memory.getPtr() != null;
    register();
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }

  public static long evictToHeap(int deviceId) {
    return CudaSystem.withDevice(deviceId, gpu -> {
      long size = RegisteredObjectBase.getLivingInstances(CudaTensorList.class)
          .filter(x -> x.gpuCopy != null && (x.getDeviceId() == deviceId || deviceId < 0 || x.getDeviceId() < 0))
          .mapToLong(CudaTensorList::evictToHeap).sum();
      logger.info(String.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
      return size;
    });
  }

  @Nonnull
  public static CudaTensorList wrap(@Nonnull final CudaTensor ptr, final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    @Nonnull CudaTensorList cudaTensorList = new CudaTensorList(ptr, length, dimensions, precision);
    ptr.freeRef();
    return cudaTensorList;
  }

  public static CudaTensorList create(final CudaTensor ptr, final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    return new CudaTensorList(ptr, length, dimensions, precision);
  }

  public static CudaTensorList create(final CudaMemory ptr, CudaDevice.CudaTensorDescriptor descriptor, final int length, @Nonnull final int[] dimensions, @Nonnull final Precision precision) {
    CudaTensor cudaTensor = new CudaTensor(ptr, descriptor, precision);
    CudaTensorList cudaTensorList = new CudaTensorList(cudaTensor, length, dimensions, precision);
    cudaTensor.freeRef();
    return cudaTensorList;
  }

  @Override
  public CudaTensorList addRef() {
    return (CudaTensorList) super.addRef();
  }

  public int getDeviceId() {
    CudaTensor gpuCopy = this.gpuCopy;
    return null == gpuCopy ? -1 : gpuCopy.memory.getDeviceId();
  }

  @Override
  public TensorList addAndFree(@Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    if (right instanceof ReshapedTensorList) return addAndFree(((ReshapedTensorList) right).getInner());
    if (1 < currentRefCount()) {
      TensorList sum = add(right);
      freeRef();
      return sum;
    }
    assert length() == right.length();
    if (heapCopy == null) {
      if (right instanceof CudaTensorList) {
        @Nonnull final CudaTensorList nativeRight = (CudaTensorList) right;
        if (nativeRight.getPrecision() == this.getPrecision()) {
          if (nativeRight.heapCopy == null) {
            assert (nativeRight.gpuCopy != this.gpuCopy);
            if (null != this.gpuCopy.memory) {
              int deviceId = this.gpuCopy.memory.getDeviceId();
              return CudaSystem.run(gpu -> {
                if (gpu.getDeviceId() == deviceId) {
                  return gpu.addInPlace(this, nativeRight);
                } else {
                  assertAlive();
                  right.assertAlive();
                  TensorList add = add(right);
                  freeRef();
                  return add;
                }
              }, this, right);
            }
          }
        }
      }
    }
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      Tensor a = get(i);
      Tensor b = right.get(i);
      @Nullable Tensor r = a.addAndFree(b);
      b.freeRef();
      return r;
    }).toArray(i -> new Tensor[i]));
  }

  @Override
  public TensorList add(@Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    assert length() == right.length();
    if (right instanceof ReshapedTensorList) return add(((ReshapedTensorList) right).getInner());
    if (heapCopy == null) {
      if (right instanceof CudaTensorList) {
        @Nonnull final CudaTensorList nativeRight = (CudaTensorList) right;
        if (nativeRight.getPrecision() == this.getPrecision()) {
          if (nativeRight.heapCopy == null) {
            return CudaSystem.run(gpu -> {
              return gpu.add(this, nativeRight);
            }, this);
          }
        }
      }
    }
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      Tensor a = get(i);
      Tensor b = right.get(i);
      @Nullable Tensor r = a.addAndFree(b);
      b.freeRef();
      return r;
    }).toArray(i -> new Tensor[i]));
  }

  @Override
  @Nonnull
  public Tensor get(final int i) {
    assertAlive();
    if (heapCopy != null) return heapCopy.get(i);
    CudaTensor gpuCopy = this.gpuCopy.addRef();
    Tensor result = CudaSystem.run(gpu -> {
      try {
        TimedResult<Tensor> timedResult = TimedResult.time(() -> {
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          Tensor t = new Tensor(getDimensions());
          if (gpuCopy.isDense()) {
            CudaMemory memory = gpuCopy.getMemory(gpu);
            memory.read(gpuCopy.getPrecision(), t.getData(), i * Tensor.length(getDimensions()));
            memory.freeRef();
          } else {
            gpuCopy.read(gpu, i, t, false);
          }
          return t;
        });
        CudaTensorList.logger.debug(String.format("Read %s bytes in %.4f from Tensor %s, GPU at %s, created by %s",
            gpuCopy.size(), timedResult.seconds(), Integer.toHexString(System.identityHashCode(timedResult.result)),
            Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
            Util.toString(createdBy).replaceAll("\n", "\n\t")));
        Tensor tensor = timedResult.result;
        return tensor;
      } finally {
        gpuCopy.freeRef();
      }
    }, CudaTensorList.this);
    result.addRef();
    result.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public int[] getDimensions() {
    return Arrays.copyOf(dimensions, dimensions.length);
  }

  /**
   * The Precision.
   */
  @Nonnull
  public Precision getPrecision() {
    CudaTensor gpuCopy = this.gpuCopy;
    return null == gpuCopy ? Precision.Double : gpuCopy.getPrecision();
  }

  @Nullable
  private TensorArray heapCopy() {
    return heapCopy(false);
  }

  @Nullable
  private TensorArray heapCopy(final boolean avoidAllocations) {
    TensorArray heapCopy = this.heapCopy;
    if (null == heapCopy || heapCopy.isFinalized()) {
      assertAlive();
      TensorArray copy = toHeap(avoidAllocations);
      final TensorArray prev;
      synchronized (this) {
        heapCopy = this.heapCopy;
        if (copy == heapCopy) {
          prev = null;
        } else if (null == heapCopy || heapCopy.isFinalized()) {
          prev = this.heapCopy;
          this.heapCopy = copy;
          heapCopy = copy;
        } else {
          copy.freeRef();
          prev = null;
        }
        if (null != prev) prev.freeRef();
      }
    }
    return heapCopy;
  }

  private TensorArray toHeap(final boolean avoidAllocations) {
    CudaTensor gpuCopy = this.gpuCopy;
    if (null == gpuCopy || !gpuCopy.tryAddRef()) {
      if (null == heapCopy) {
        throw new IllegalStateException("No data");
      } else if (heapCopy.isFinalized()) {
        throw new IllegalStateException("Local data has been freed");
      } else {
        return heapCopy;
      }
    }
    int length = getLength();
    if (0 >= length) throw new IllegalStateException();
    final Tensor[] output = IntStream.range(0, length)
        .mapToObj(dataIndex -> new Tensor(getDimensions()))
        .toArray(i -> new Tensor[i]);
    TimedResult<TensorArray> timedResult = TimedResult.time(() -> CudaDevice.run(gpu -> {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      try {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
//        assert getPrecision() == gpuCopy.getPrecision();
//        assert getPrecision() == gpuCopy.descriptor.dataType;
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        for (int i = 0; i < length; i++) {
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          gpuCopy.read(gpu, i, output[i], avoidAllocations);
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        }
        return TensorArray.wrap(output);
      } finally {
        if (gpuCopy != null) gpuCopy.freeRef();
      }
    }, this));
    CudaTensorList.logger.debug(String.format("Read %s bytes in %.4f from Tensor %s on GPU at %s, created by %s",
        gpuCopy.size(), timedResult.seconds(), Integer.toHexString(System.identityHashCode(timedResult.result)),
        Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
        Util.toString(createdBy).replaceAll("\n", "\n\t")));
    return timedResult.result;
  }


  @Override
  public int length() {
    return getLength();
  }

  @Override
  public Stream<Tensor> stream() {
    return heapCopy().stream();
  }

  @Override
  public TensorList copy() {
    return CudaSystem.run(gpu -> {
      CudaTensor ptr = gpu.getTensor(this, MemoryType.Device, false);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      CudaMemory cudaMemory = ptr.getMemory(gpu, MemoryType.Device);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      CudaMemory copyPtr = cudaMemory.copy(gpu, MemoryType.Managed.ifEnabled());
      cudaMemory.freeRef();
      try {
        CudaTensor cudaTensor = new CudaTensor(copyPtr, ptr.descriptor, getPrecision());
        CudaTensorList cudaTensorList = new CudaTensorList(cudaTensor, getLength(), getDimensions(), getPrecision());
        cudaTensor.freeRef();
        return cudaTensorList;
      } finally {
        copyPtr.freeRef();
        ptr.freeRef();
      }
    }, this);
  }

  @Override
  protected void _free() {
    synchronized (this) {
      if (null != gpuCopy) {
        gpuCopy.freeRef();
        gpuCopy = null;
      }
      if (null != heapCopy) {
        heapCopy.freeRef();
        heapCopy = null;
      }
    }
  }

  public long evictToHeap() {
    if (isFinalized()) return 0;
    if (null == heapCopy(true)) {
      throw new IllegalStateException();
    }
    CudaTensor ptr;
    synchronized (this) {
      ptr = this.gpuCopy;
      this.gpuCopy = null;
    }
    if (null != ptr && !ptr.isFinalized()) {
      long elements = getElements();
      assert 0 < length;
      assert 0 < elements : Arrays.toString(dimensions);
      return 0 == ptr.freeRef() ? (elements * getPrecision().size) : 0;
    } else {
      synchronized (this) {
        if (null != this.gpuCopy) this.gpuCopy.freeRef();
        this.gpuCopy = ptr;
      }
      return 0;
    }
  }

  public int getLength() {
    return length;
  }
}
