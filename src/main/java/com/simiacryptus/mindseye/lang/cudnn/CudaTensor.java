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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefConsumer;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.function.Function;

public @RefAware
class CudaTensor extends ReferenceCountingBase implements CudaSystem.CudaDeviceResource {
  static final Logger log = LoggerFactory.getLogger(CudaTensor.class);

  public final CudaDevice.CudaTensorDescriptor descriptor;
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE().isProfileMemoryIO() ? Util.getStackTrace()
      : new StackTraceElement[]{};
  final CudaMemory memory;

  public CudaTensor(final CudaMemory memory, final CudaDevice.CudaTensorDescriptor descriptor,
                    final Precision precision) {
    {
      CudaMemory temp_18_0001 = memory == null ? null : memory.addRef();
      this.memory = temp_18_0001 == null ? null : temp_18_0001.addRef();
      if (null != temp_18_0001)
        temp_18_0001.freeRef();
    }
    {
      CudaDevice.CudaTensorDescriptor temp_18_0002 = descriptor == null ? null
          : descriptor.addRef();
      this.descriptor = temp_18_0002 == null ? null : temp_18_0002.addRef();
      if (null != temp_18_0002)
        temp_18_0002.freeRef();
    }
    assert memory.size >= (long) precision.size * descriptor.nStride * (descriptor.batchCount - 1) : String
        .format("%d != %d", memory.size, (long) precision.size * descriptor.nStride * descriptor.batchCount);
    if (null != memory)
      memory.freeRef();
    assert this.descriptor.dataType == precision;
    if (null != descriptor)
      descriptor.freeRef();
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

  public static @SuppressWarnings("unused")
  CudaTensor[] addRefs(CudaTensor[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaTensor::addRef)
        .toArray((x) -> new CudaTensor[x]);
  }

  public static @SuppressWarnings("unused")
  CudaTensor[][] addRefs(CudaTensor[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaTensor::addRefs)
        .toArray((x) -> new CudaTensor[x][]);
  }

  public CudaMemory getMemory(final CudaDevice cudaDevice) {
    return getMemory(cudaDevice, MemoryType.Device);
  }

  public CudaMemory getMemory(final CudaDevice cudaDevice, final MemoryType memoryType) {
    assertAlive();
    //    memory.synchronize();
    if (memory.getType() == MemoryType.Managed) {
      return memory == null ? null : memory.addRef();
    } else if (cudaDevice.getDeviceId() == memory.getDeviceId()) {
      return memory == null ? null : memory.addRef();
    } else {
      TimedResult<CudaMemory> timedResult = TimedResult.time(() -> memory.copy(cudaDevice, memoryType));
      CudaTensorList.logger
          .debug(RefString.format("Copy %s bytes in %.4f from Tensor %s on GPU %s to %s at %s, created by %s", memory.size,
              timedResult.seconds(), Integer.toHexString(com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(this)), memory.getDeviceId(),
              cudaDevice.getDeviceId(), Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
              Util.toString(createdBy).replaceAll("\n", "\n\t")));
      return timedResult.result;
    }
  }

  public CudaTensor getDense(CudnnHandle gpu) {
    assertAlive();
    if (isDense()) {
      return this.addRef();
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
      RefUtil.freeRef(memory.dirty());
      if (null != memory)
        memory.freeRef();
      RefUtil.freeRef(destMemory.dirty());
      CudaTensor temp_18_0003 = new CudaTensor(
          destMemory == null ? null : destMemory.addRef(), destDescriptor == null ? null : destDescriptor.addRef(),
          getPrecision());
      if (null != destMemory)
        destMemory.freeRef();
      if (null != destDescriptor)
        destDescriptor.freeRef();
      return temp_18_0003;
    });
    CudaTensor cudaTensor = timedResult.result.addRef();
    assert cudaTensor.isDense();
    CudaTensorList.logger.debug(RefString.format("Densified %s bytes in %.4f from GPU %s at %s, created by %s",
        cudaTensor.memory.size, timedResult.seconds(), Integer.toHexString(com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(this)),
        Util.toString(Util.getStackTrace()).replaceAll("\n", "\n\t"),
        Util.toString(createdBy).replaceAll("\n", "\n\t")));
    return cudaTensor;

  }

  public void read(final CudnnHandle gpu, final int index, final Tensor result, final boolean avoidAllocations) {
    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
    if (isDense()) {
      try {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        CudaSystem.withDevice(memory.getDeviceId(), RefUtil.wrapInterface(
            (RefConsumer<CudnnHandle>) dev -> {
              assert CudaDevice.isThreadDeviceId(dev.getDeviceId());
              CudaMemory memory = getMemory(dev);
              RefUtil
                  .freeRef(memory.read(descriptor.dataType, result.getData(), index * descriptor.nStride));
              if (null != memory)
                memory.freeRef();
              assert CudaDevice.isThreadDeviceId(dev.getDeviceId());
            }, result == null ? null : result.addRef()));
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
          RefUtil
              .freeRef(memory.read(descriptor.dataType, buffer, descriptor.nStride * index));
          RefUtil.freeRef(result.setByCoord(c -> {
            int[] coords = c.getCoords();
            int x = coords.length < 1 ? 1 : coords[0];
            int y = coords.length < 2 ? 1 : coords[1];
            int z = coords.length < 3 ? 1 : coords[2];
            return buffer[x * descriptor.wStride + y * descriptor.hStride + z * descriptor.cStride];
          }));
        } finally {
          RecycleBin.DOUBLES.recycle(buffer, buffer.length);
        }
      } finally {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      }
    } else {
      try {
        withDense(gpu, index, RefUtil.wrapInterface(
            (Function<CudaMemory, CudaMemory>) mem -> {
              CudaMemory temp_18_0004 = mem.read(this.descriptor.dataType,
                  result.getData());
              if (null != mem)
                mem.freeRef();
              return temp_18_0004;
            }, result == null ? null : result.addRef()));
      } finally {
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      }
    }
    if (null != result)
      result.freeRef();
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
          RefUtil.freeRef(memory.dirty());
          RefUtil.freeRef(cudaMemory.dirty());
          if (null != sourceDescriptor)
            sourceDescriptor.freeRef();
          if (null != destDescriptor)
            destDescriptor.freeRef();
          T temp_18_0005 = result.apply(cudaMemory);
          if (null != cudaMemory)
            cudaMemory.freeRef();
          if (null != memory)
            memory.freeRef();
          return temp_18_0005;
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
    if (null != memory)
      memory.freeRef();
    if (null != descriptor)
      descriptor.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  CudaTensor addRef() {
    return (CudaTensor) super.addRef();
  }
}
