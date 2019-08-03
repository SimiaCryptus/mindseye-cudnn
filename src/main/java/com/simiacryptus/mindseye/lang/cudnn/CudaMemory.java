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

import com.simiacryptus.mindseye.lang.RegisteredObjectBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import jcuda.runtime.cudaMemcpyKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CudaMemory extends CudaResourceBase<CudaPointer> {

  public static final Map<Integer, DeviceMetrics> METRICS = new ConcurrentHashMap<>();
  public static final int K = 1024;
  public static final long MiB = K * 1024;
  public static final long GiB = 1024 * MiB;
  protected static final Logger logger = LoggerFactory.getLogger(CudaMemory.class);
  public final long size;
  private final int deviceId;
  private final MemoryType type;
  private long writtenAt = System.nanoTime();

  CudaMemory(final CudaDevice gpu, final long size, @Nonnull MemoryType type) {
    this(size, type, gpu.acquire(size, type, 1), gpu.getDeviceId());
  }

  CudaMemory(final long size, @Nonnull MemoryType type, final CudaPointer memory, final int deviceId) {
    super(memory);
    this.size = size;
    this.deviceId = deviceId;
    this.type = type;
  }

  public static double clearWeakMemory(final int deviceId) {
    double totalFreed = 0;
    for (final MemoryType type : MemoryType.values()) {
      totalFreed += type.purge(deviceId);
    }
    return totalFreed;
  }

  public static double clearMemory(final int deviceId) {
    double totalFreed = evictMemory(deviceId);
    for (final MemoryType type : MemoryType.values()) {
      totalFreed += type.purge(deviceId);
    }
    if (totalFreed == 0) {
      logger.info(String.format("Nothing Freed - Running Garbage Collector"));
      System.gc();
      totalFreed = evictMemory(0);
    }
    if (totalFreed == 0) {
      logger.info(String.format("Warning: High Active GPU Memory Usage"));
    }
    logLoad();
    return totalFreed;
  }

  public static double evictMemory(final int deviceId) {
    double bytes = RegisteredObjectBase.getLivingInstances(SimpleConvolutionLayer.class).mapToLong(x -> x.evictDeviceData(deviceId)).sum();
    logger.info(String.format("Cleared %e bytes from ConvolutionFilters for device %s", bytes, deviceId));
    double tensorListsFreed = CudaTensorList.evictToHeap(deviceId);
    return tensorListsFreed + bytes;
  }

  private static void logLoad() {
    logger.info(String.format("Current Load: %s", METRICS.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> {
      return String.format("%e / %e", (double) e.getValue().activeMemory.get(), (double) e.getValue().usedMemory.get());
    }))));
  }


  public static DeviceMetrics getGpuStats(final int deviceId) {
    return CudaMemory.METRICS.computeIfAbsent(deviceId, device -> new DeviceMetrics());
  }

  @Nonnull
  public Tensor read(@Nonnull final Precision precision, final int[] dimensions) {
    synchronize();
    @Nonnull final Tensor tensor = new Tensor(dimensions);
    switch (precision) {
      case Float:
        final int length = tensor.length();
        @Nonnull final float[] data = new float[length];
        read(precision, data);
        @Nullable final double[] doubles = tensor.getData();
        for (int i = 0; i < length; i++) {
          doubles[i] = data[i];
        }
        break;
      case Double:
        read(precision, tensor.getData());
        break;
      default:
        throw new IllegalStateException();
    }
    return tensor;
  }

  public CudaMemory copy(CudaDevice deviceId, final MemoryType memoryType) {
    @Nonnull CudaMemory copy = deviceId.allocate(size, memoryType, true);
    synchronize();
    CudaSystem.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    return copy;
  }


  protected void _free() {
    if (ptr.getByteOffset() != 0) return;
    CudnnHandle threadHandle = CudaSystem.getThreadHandle();
    if (null != threadHandle) threadHandle.cleanupNative.add(this);
    else release();
  }

  @Override
  public void release() {
    if (ptr.getByteOffset() != 0) return;
    if (isActiveObj()) {
      synchronize();
      getType().recycle(ptr, deviceId, size);
      ptr = null;
      CudaMemory.getGpuStats(deviceId).activeMemory.addAndGet(-size);
    }
  }

  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final double[] destination) {
    return read(precision, destination, 0);
  }

  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final double[] destination, int offset) {
    if (0 == destination.length) return this;
    if (size < (long) (offset + destination.length) * precision.size) {
      throw new IllegalArgumentException(String.format("%d < %d + %d", size, (long) destination.length * precision.size, offset));
    }
    if (precision == Precision.Float) {
      @Nonnull float[] data = new float[destination.length];
      read(Precision.Float, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = data[i];
      }
    } else {
      synchronize();
      CudaSystem.run(gpu -> {
        CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size), (long) destination.length * precision.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
      });
      CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) destination.length * precision.size);
    }
    return this;
  }

  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final float[] destination) {
    return read(precision, destination, 0);
  }

  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final float[] destination, int offset) {
    if (size < (long) destination.length * precision.size) {
      throw new IllegalArgumentException(size + " != " + (long) destination.length * precision.size);
    }
    if (precision == Precision.Double) {
      @Nonnull double[] data = new double[destination.length];
      read(Precision.Double, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = (float) data[i];
      }
    } else {
      synchronize();
      CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size), (long) destination.length * precision.size, cudaMemcpyDeviceToHost);
      CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) destination.length * precision.size);
    }
    return this;
  }

  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final double[] data) {
    return write(precision, data, 0);
  }

  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final double[] data, long offset) {
    assert getType() == MemoryType.Managed || CudaDevice.isThreadDeviceId(getDeviceId());
    if (size < ((offset + data.length) * precision.size))
      throw new IllegalArgumentException(String.format("%d != (%d + %d) * %d", size, offset, data.length, precision.size));
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data), (long) data.length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) data.length * precision.size);
    return this;
  }

  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final float[] data) {
    return write(precision, data, 0);
  }

  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final float[] data, int offset) {
    if (size < (offset + data.length) * precision.size)
      throw new IllegalArgumentException(String.format("%d != %d * %d", size, data.length, precision.size));
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data), (long) data.length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) data.length * precision.size);
    return this;
  }

  public int getDeviceId() {
    return deviceId;
  }

  @Nonnull
  public MemoryType getType() {
    return type;
  }

  @Nonnull
  CudaMemory clear() {
    CudaSystem.cudaMemset(getPtr(), 0, size);
    return this;
  }

  public CudaMemory withByteOffset(final int byteOffset) {
    if (size <= byteOffset) throw new IllegalArgumentException(size + " <= " + byteOffset);
    if (0 > byteOffset) throw new IllegalArgumentException(Integer.toString(byteOffset));
    assertAlive();
    final CudaMemory baseMemorySegment = this;
    baseMemorySegment.addRef();
    return new CudaMemory(size - byteOffset, type, ptr.withByteOffset(byteOffset), baseMemorySegment.getDeviceId()) {
      @Override
      protected void _free() {
        baseMemorySegment.freeRef();
      }

      @Override
      public void release() {
      }
    };
  }

  public CudaMemory dirty() {
    assert type == MemoryType.Managed || CudaDevice.isThreadDeviceId(getDeviceId()) : getDeviceId() + " != " + CudaSystem.getThreadDeviceId();
    writtenAt = System.nanoTime();
    return this;
  }

  public void synchronize() {
    if (deviceId >= 0) CudaSystem.synchronize(writtenAt, deviceId);
  }

  @Override
  public CudaMemory addRef() {
    return (CudaMemory) super.addRef();
  }

}
