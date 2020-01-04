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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.function.Function;

public @com.simiacryptus.ref.lang.RefAware class CudaTensor extends ReferenceCountingBase
    implements CudaSystem.CudaDeviceResource {
  static final Logger log = LoggerFactory.getLogger(CudaTensor.class);

  public final CudaDevice.CudaTensorDescriptor descriptor;
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace()
      : new StackTraceElement[] {};
  final CudaMemory memory;

  public CudaTensor(final CudaMemory memory, final CudaDevice.CudaTensorDescriptor descriptor,
      final Precision precision) {
    this.memory = memory;
    this.descriptor = descriptor;
    assert memory.size >= (long) precision.size * descriptor.nStride * (descriptor.batchCount - 1) : String
        .format("%d != %d", memory.size, (long) precision.size * descriptor.nStride * descriptor.batchCount);
    assert this.descriptor.dataType == precision;
  }

  public int getDeviceId() {
    return memory.getDeviceId();
  }

  public Precision getPrecision() {
    return descriptor.dataType;
  }

  public MemoryType getType() {
    return memory.getType();
  }

  public boolean isDense() {
    if (descriptor.nStride != descriptor.channels * descriptor.height * descriptor.width)
      return false;
    if (descriptor.cStride != descriptor.height * descriptor.width)
      return false;
    if (descriptor.hStride != descriptor.width)
      return false;
    return descriptor.wStride == 1;
  }

  public CudaMemory getMemory(final CudaDevice cudaDevice) {
    return getMemory(cudaDevice, MemoryType.Device);
  }

  public CudaMemory getMemory(final CudaDevice cudaDevice, final MemoryType memoryType) {
    assertAlive();
    //    memory.synchronize();
    if (memory.getType() == MemoryType.Managed) {
      return memory;
    } else if (cudaDevice.getDeviceId() == memory.getDeviceId()) {
      return memory;
    } else {
      TimedResult<CudaMemory> timedResult = TimedResult.time(() -> memory.copy(cudaDevice, memoryType));
      CudaTensorList.logger
          .debug(String.format("Copy %s bytes in %.4f from Tensor %s on GPU %s to %s at %s, created by %s", memory.size,
              timedResult.seconds(), Integer.toHexString(System.identityHashCode(this)), memory.getDeviceId(),
              cudaDevice.getDeviceId(), Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
              Util.toString(createdBy).replaceAll("\n", "\n\t")));
      return timedResult.result;
    }
  }

  public CudaTensor getDense(CudnnHandle gpu) {
    assertAlive();
    if (isDense()) {
      return this;
    }
    TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
      CudaDevice.CudaTensorDescriptor destDescriptor = gpu.newTensorDescriptor(getPrecision(),
          this.descriptor.batchCount, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
          this.descriptor.channels * this.descriptor.height * this.descriptor.width,
          this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
      CudaMemory destMemory = gpu.allocate(destDescriptor.nStride * destDescriptor.batchCount * getPrecision().size,
          getType(), true);
      CudaMemory memory = getMemory(gpu);
      gpu.cudnnTransformTensor(getPrecision().getPointer(1.0), this.descriptor.getPtr(), memory.getPtr(),
          getPrecision().getPointer(0.0), destDescriptor.getPtr(), destMemory.getPtr());
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      memory.dirty();
      destMemory.dirty();
      return new CudaTensor(destMemory, destDescriptor, getPrecision());
    });
    CudaTensor cudaTensor = timedResult.result;
    assert cudaTensor.isDense();
    CudaTensorList.logger.debug(String.format("Densified %s bytes in %.4f from GPU %s at %s, created by %s",
        cudaTensor.memory.size, timedResult.seconds(), Integer.toHexString(System.identityHashCode(this)),
        Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
        Util.toString(createdBy).replaceAll("\n", "\n\t")));
    return cudaTensor;

  }

  public void read(final CudnnHandle gpu, final int index, final Tensor result, final boolean avoidAllocations) {
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    if (isDense()) {
      try {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        CudaSystem.withDevice(memory.getDeviceId(), dev -> {
          assert CudaDevice.isThreadDeviceId(dev.getDeviceId());
          CudaMemory memory = getMemory(dev);
          memory.read(descriptor.dataType, result.getData(), index * descriptor.nStride);
          assert CudaDevice.isThreadDeviceId(dev.getDeviceId());
        });
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      } catch (Throwable e) {
        log.warn("Error", e);
      } finally {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      }
    } else if (avoidAllocations) {
      try {
        int size = (descriptor.channels - 1) * descriptor.cStride + (descriptor.height - 1) * descriptor.hStride
            + (descriptor.width - 1) * descriptor.wStride + 1;
        double[] buffer = RecycleBin.DOUBLES.obtain(size);
        try {
          memory.read(descriptor.dataType, buffer, descriptor.nStride * index);
          result.setByCoord(c -> {
            int[] coords = c.getCoords();
            int x = coords.length < 1 ? 1 : coords[0];
            int y = coords.length < 2 ? 1 : coords[1];
            int z = coords.length < 3 ? 1 : coords[2];
            return buffer[x * descriptor.wStride + y * descriptor.hStride + z * descriptor.cStride];
          });
        } finally {
          RecycleBin.DOUBLES.recycle(buffer, buffer.length);
        }
      } finally {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      }
    } else {
      try {
        withDense(gpu, index, mem -> mem.read(this.descriptor.dataType, result.getData()));
      } finally {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      }
    }
  }

  @Nonnull
  public <T> void withDense(final CudnnHandle gpu, final int index, final Function<CudaMemory, T> result) {
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    int deviceId = memory.getDeviceId();
    Function<CudnnHandle, T> fn = dev -> {
      assertAlive();
      assert this.descriptor.dataType == getPrecision();
      CudaDevice.CudaTensorDescriptor sourceDescriptor = dev.newTensorDescriptor(this.descriptor.dataType, 1,
          this.descriptor.channels, this.descriptor.height, this.descriptor.width, this.descriptor.nStride,
          this.descriptor.cStride, this.descriptor.hStride, this.descriptor.wStride);
      CudaDevice.CudaTensorDescriptor destDescriptor = dev.newTensorDescriptor(this.descriptor.dataType, 1,
          this.descriptor.channels, this.descriptor.height, this.descriptor.width,
          this.descriptor.channels * this.descriptor.height * this.descriptor.width,
          this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
      try {
        CudaMemory cudaMemory = dev.allocate(destDescriptor.nStride * this.descriptor.dataType.size, MemoryType.Device,
            true);
        CudaMemory memory = getMemory(dev);
        {
          dev.cudnnTransformTensor(this.descriptor.dataType.getPointer(1.0), sourceDescriptor.getPtr(),
              memory.getPtr().withByteOffset(index * this.descriptor.nStride * getPrecision().size),
              this.descriptor.dataType.getPointer(0.0), destDescriptor.getPtr(), cudaMemory.getPtr());
          memory.dirty();
          cudaMemory.dirty();
          return result.apply(cudaMemory);
        }
      } finally {
        assert CudaDevice.isThreadDeviceId(dev.getDeviceId());
      }
    };
    try {
      if (0 > deviceId || gpu.getDeviceId() == deviceId) {
        fn.apply(gpu);
      } else {
        CudaSystem.withDevice(deviceId, fn);
      }
    } finally {
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    }
  }

  public long size() {
    return memory.size;
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") CudaTensor addRef() {
    return (CudaTensor) super.addRef();
  }

  public static @SuppressWarnings("unused") CudaTensor[] addRefs(CudaTensor[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CudaTensor::addRef)
        .toArray((x) -> new CudaTensor[x]);
  }

  public static @SuppressWarnings("unused") CudaTensor[][] addRefs(CudaTensor[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CudaTensor::addRefs)
        .toArray((x) -> new CudaTensor[x][]);
  }
}
