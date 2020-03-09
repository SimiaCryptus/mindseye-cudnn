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

import com.simiacryptus.mindseye.lang.ObjectRegistry;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import jcuda.runtime.cudaMemcpyKind;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CudaMemory extends CudaResourceBase<CudaPointer> {

  public static final RefMap<Integer, DeviceMetrics> METRICS = new RefConcurrentHashMap<>();
  public static final int K = 1024;
  public static final long MiB = K * 1024;
  public static final long GiB = 1024 * MiB;
  protected static final Logger logger = LoggerFactory.getLogger(CudaMemory.class);
  public final long size;
  private final int deviceId;
  @Nonnull
  private final MemoryType type;
  private long writtenAt = RefSystem.nanoTime();

  CudaMemory(@Nonnull final CudaDevice gpu, final long size, @Nonnull MemoryType type) {
    this(size, type, gpu.acquire(size, type, 1), gpu.getDeviceId());
    gpu.freeRef();
  }

  CudaMemory(final long size, @Nonnull MemoryType type, final CudaPointer memory, final int deviceId) {
    super(memory);
    this.size = size;
    this.deviceId = deviceId;
    this.type = type;
  }

  public int getDeviceId() {
    return deviceId;
  }

  @Nonnull
  public MemoryType getType() {
    return type;
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
      logger.info(RefString.format("Nothing Freed - Running Garbage Collector"));
      RefSystem.gc();
      totalFreed = evictMemory(0);
    }
    if (totalFreed == 0) {
      logger.info(RefString.format("Warning: High Active GPU Memory Usage"));
    }
    logLoad();
    return totalFreed;
  }

  public static double evictMemory(final int deviceId) {
    double bytes = ObjectRegistry.getLivingInstances(SimpleConvolutionLayer.class).mapToLong(x -> {
      long temp_35_0001 = x.evictDeviceData(deviceId);
      x.freeRef();
      return temp_35_0001;
    }).sum();
    logger.debug(RefString.format("Cleared %e bytes from ConvolutionFilters for device %s", bytes, deviceId));
    double tensorListsFreed = CudaTensorList.evictToHeap(deviceId);
    return tensorListsFreed + bytes;
  }

  @NotNull
  public static DeviceMetrics getGpuStats(final int deviceId) {
    return CudaMemory.METRICS.computeIfAbsent(deviceId, device -> new DeviceMetrics());
  }


  private static void logLoad() {
    RefSet<Map.Entry<Integer, DeviceMetrics>> temp_35_0005 = METRICS.entrySet();
    RefMap<Integer, String> temp_35_0006 = temp_35_0005.stream().collect(RefCollectors.toMap(e -> {
      Integer temp_35_0002 = e.getKey();
      RefUtil.freeRef(e);
      return temp_35_0002;
    }, e -> {
      String temp_35_0003 = RefString.format("%e / %e", (double) e.getValue().activeMemory.get(),
          (double) e.getValue().usedMemory.get());
      RefUtil.freeRef(e);
      return temp_35_0003;
    }));
    logger.debug(RefString.format("Current Load: %s", temp_35_0006));
    temp_35_0005.freeRef();
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
        tensor.set(i -> data[i]);
        return tensor;
      case Double:
        read(precision, tensor.addRef(), 0);
        return tensor;
      default:
        tensor.freeRef();
        throw new IllegalStateException();
    }
  }

  @Nonnull
  public CudaMemory copy(@Nonnull CudaDevice deviceId, @Nonnull final MemoryType memoryType) {
    @Nonnull
    CudaMemory copy = deviceId.allocate(size, memoryType, true);
    deviceId.freeRef();
    synchronize();
    CudaSystem.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    return copy;
  }

  @Override
  public void release() {
    assert ptr != null;
    if (ptr.getByteOffset() != 0)
      return;
    if (isActiveObj()) {
      synchronize();
      getType().recycle(ptr, deviceId, size);
      ptr = null;
      CudaMemory.getGpuStats(deviceId).activeMemory.addAndGet(-size);
    }
  }

  public void read(@Nonnull Precision precision, @Nonnull Tensor destination, int offset) {
    int length = destination.length();
    if (0 != length) {
      if (size < (long) (offset + length) * precision.size) {
        destination.freeRef();
        throw new IllegalArgumentException(
            RefString.format("%d < %d + %d", size, (long) length * precision.size, offset));
      }
      if (precision == Precision.Float) {
        @Nonnull
        float[] data = new float[length];
        read(Precision.Float, data, offset);
        destination.set(i -> data[i]);
      } else {
        synchronize();
        CudaSystem.run(gpu -> {
          CudaSystem.cudaMemcpy(precision.getPointer(destination.getData()),
              getPtr().withByteOffset((long) offset * precision.size), (long) length * precision.size,
              cudaMemcpyKind.cudaMemcpyDeviceToHost);
          gpu.freeRef();
        });
        CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) length * precision.size);
      }
    }
    destination.freeRef();
  }

  public void read(@Nonnull Precision precision, @Nonnull double[] destination, int offset) {
    int length = destination.length;
    if (0 != length) {
      if (size < (long) (offset + length) * precision.size) {
        throw new IllegalArgumentException(
            RefString.format("%d < %d + %d", size, (long) length * precision.size, offset));
      }
      if (precision == Precision.Float) {
        @Nonnull
        float[] data = new float[length];
        read(Precision.Float, data, offset);
        for (int i = 0; i < length; i++) {
          destination[i] = data[i];
        }
      } else {
        synchronize();
        CudaSystem.run(gpu -> {
          CudaSystem.cudaMemcpy(precision.getPointer(destination),
              getPtr().withByteOffset((long) offset * precision.size), (long) length * precision.size,
              cudaMemcpyKind.cudaMemcpyDeviceToHost);
          gpu.freeRef();
        });
        CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) length * precision.size);
      }
    }
  }

  @Nonnull
  public void read(@Nonnull final Precision precision, @Nonnull final float[] destination) {
    read(precision, destination, 0);
  }

  @Nonnull
  public void read(@Nonnull final Precision precision, @Nonnull final float[] destination, int offset) {
    if (size < (long) destination.length * precision.size) {
      throw new IllegalArgumentException(size + " != " + (long) destination.length * precision.size);
    }
    if (precision == Precision.Double) {
      @Nonnull
      double[] data = new double[destination.length];
      read(Precision.Double, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = (float) data[i];
      }
    } else {
      synchronize();
      CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size),
          (long) destination.length * precision.size, cudaMemcpyDeviceToHost);
      CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) destination.length * precision.size);
    }
  }

  public void write(@Nonnull Precision precision, @Nonnull Tensor data) {
    write(precision, data, 0);
  }

  public void write(@Nonnull Precision precision, @Nonnull Tensor data, long offset) {
    int length = data.length();
    assert getType() == MemoryType.Managed || CudaDevice.isThreadDeviceId(getDeviceId());
    if (size < (offset + length) * precision.size) {
      data.freeRef();
      throw new IllegalArgumentException(
          RefString.format("%d != (%d + %d) * %d", size, offset, length, precision.size));
    }
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data.getData()),
        (long) length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) length * precision.size);
    data.freeRef();
  }

  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final float[] data) {
    write(precision, data, 0);
    return this.addRef();
  }

  public void write(@Nonnull Precision precision, @Nonnull float[] data, int offset) {
    if (size < (offset + data.length) * precision.size)
      throw new IllegalArgumentException(RefString.format("%d != %d * %d", size, data.length, precision.size));
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data),
        (long) data.length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) data.length * precision.size);
  }

  @Nonnull
  public CudaMemory withByteOffset(final int byteOffset) {
    if (size <= byteOffset)
      throw new IllegalArgumentException(size + " <= " + byteOffset);
    if (0 > byteOffset)
      throw new IllegalArgumentException(Integer.toString(byteOffset));
    assertAlive();
    assert ptr != null;
    return new OffsetCudaMemory(this.addRef(), byteOffset);
  }

  public void dirty() {
    assert type == MemoryType.Managed || CudaDevice.isThreadDeviceId(getDeviceId()) : getDeviceId() + " != "
        + CudaSystem.getThreadDeviceId();
    writtenAt = RefSystem.nanoTime();
  }

  public void synchronize() {
    if (deviceId >= 0)
      CudaSystem.synchronize(writtenAt, deviceId);
  }

  public void _free() {
    super._free();
    assert ptr != null;
    if (ptr.getByteOffset() != 0) return;
    cleanup();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CudaMemory addRef() {
    return (CudaMemory) super.addRef();
  }

  @Nonnull
  void clear() {
    CudaSystem.cudaMemset(getPtr(), 0, size);
  }

  public static class OffsetCudaMemory extends CudaMemory {
    private final CudaMemory base;

    public OffsetCudaMemory(CudaMemory base, int byteOffset) {
      super(base.size - byteOffset, base.type, base.ptr.withByteOffset(byteOffset), base.getDeviceId());
      this.base = base;
    }

    @Override
    public void release() {
    }

    public void _free() {
      base.freeRef();
      super._free();
    }
  }
}
