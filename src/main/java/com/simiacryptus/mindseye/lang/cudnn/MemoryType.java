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

import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceWrapper;
import com.simiacryptus.ref.wrappers.RefConcurrentHashMap;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.Util;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import static jcuda.runtime.JCuda.*;

public enum MemoryType {

  Managed {
    @Nonnull
    public CudaPointer alloc(final long size, @Nonnull final CudaDevice cudaDevice) {
      cudaDevice.freeRef();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > (double) CudaSettings.INSTANCE().maxAllocSize) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      CudaPointer pointer = new CudaPointer();
      CudaSystem.handle(CudaSystem.cudaMallocManaged(pointer, size, cudaMemAttachGlobal));
      return pointer;
    }

    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }

    @Nonnull
    public MemoryType ifEnabled() {
      return CudaSettings.INSTANCE().enableManaged ? this : Device;
    }
  },
  Device {
    @Nonnull
    public CudaPointer alloc(final long size, @Nonnull final CudaDevice cudaDevice) {
      CudaPointer pointer = new CudaPointer();
      int returnCode;
      synchronized (cudaDevice.allocationLock) {
        cudaDevice.ensureCapacity(size);
        cudaDevice.freeRef();
        returnCode = CudaSystem.cudaMalloc(pointer, size);
      }
      CudaSystem.handle(returnCode);
      return pointer;
    }

    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }
  },
  Host {
    @Nonnull
    public CudaPointer alloc(final long size, @Nonnull final CudaDevice cudaDevice) {
      cudaDevice.freeRef();
      CudaPointer pointer = new CudaPointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > (double) CudaSettings.INSTANCE().maxAllocSize) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDeviceId());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocDefault));
      } else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }

    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  },
  HostWriteable {
    @Nonnull
    public CudaPointer alloc(final long size, @Nonnull final CudaDevice cudaDevice) {
      cudaDevice.freeRef();
      CudaPointer pointer = new CudaPointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > (double) CudaSettings.INSTANCE().maxAllocSize) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDeviceId());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocWriteCombined));
      } else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }

    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  };

  protected static final Logger logger = LoggerFactory.getLogger(MemoryType.class);
  private static final RefMap<MemoryType, RefMap<Integer, RecycleBin<ReferenceWrapper<CudaPointer>>>> cache = new RefConcurrentHashMap<>();

  public void recycle(CudaPointer ptr, int deviceId, final long length) {
    logger.debug(RefString.format("Recycle %s %s (%s bytes) in device %s via %s", name(),
        Integer.toHexString(RefSystem.identityHashCode(ptr)), length, deviceId,
        CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : ""));
    get(deviceId).recycle(new ReferenceWrapper<>(ptr, x -> {
      logger.debug(RefString.format("Freed %s %s (%s bytes) in device %s via %s", name(),
          Integer.toHexString(RefSystem.identityHashCode(ptr)), length, deviceId,
          CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : ""));
      CudaMemory.getGpuStats(deviceId).usedMemory.addAndGet(-length);
      MemoryType.this.free(x, deviceId);
    }), length);
  }

  @Nonnull
  public MemoryType ifEnabled() {
    return this;
  }

  public double purge(final int device) {
    double clear = get(device).clear();
    logger.debug(RefString.format("Purged %e bytes from pool for %s (device %s)", clear, this, device));
    return clear;
  }

  public CudaPointer allocCached(final long size, @Nonnull final CudaDevice cudaDevice) {
    RecycleBin<ReferenceWrapper<CudaPointer>> recycleBin = get(cudaDevice.deviceId);
    cudaDevice.freeRef();
    assert recycleBin != null;
    ReferenceWrapper<CudaPointer> wrapper = recycleBin.obtain(size);
    return wrapper.unwrap();
  }

  @Nonnull
  public abstract CudaPointer alloc(final long size, final CudaDevice cudaDevice);

  abstract void free(CudaPointer ptr, int deviceId);

  @Nullable
  protected RecycleBin<ReferenceWrapper<CudaPointer>> get(int device) {
    RefMap<Integer, RecycleBin<ReferenceWrapper<CudaPointer>>> temp_76_0002 = cache.computeIfAbsent(this,
        x -> new RefConcurrentHashMap<>());
    RecycleBin<ReferenceWrapper<CudaPointer>> temp_76_0001 = temp_76_0002.computeIfAbsent(device, d -> {
      logger.info(RefString.format("Initialize recycle bin %s (device %s)", this, device));
      return new RecycleBin<ReferenceWrapper<CudaPointer>>() {
        @Override
        public ReferenceWrapper<CudaPointer> obtain(final long length) {
          assert -1 == device || CudaSystem.getThreadDeviceId() == device;
          ReferenceWrapper<CudaPointer> referenceWrapper = super.obtain(length);
          MemoryType.logger.debug(RefString.format("Obtained %s %s (%s bytes) in device %s via %s", name(),
              Integer.toHexString(RefSystem.identityHashCode(referenceWrapper.peek())),
              length, device, CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : ""));
          return referenceWrapper;
        }

        @Nonnull
        @Override
        public ReferenceWrapper<CudaPointer> create(final long length) {
          assert -1 == device || CudaSystem.getThreadDeviceId() == device;
          CharSequence caller = CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : "";
          return CudaDevice.run(gpu -> {
            CudaPointer alloc = alloc(length, gpu);
            MemoryType.logger.debug(RefString.format("Created %s %s (%s bytes) in device %s via %s", name(),
                Integer.toHexString(RefSystem.identityHashCode(alloc)), length, device,
                caller));
            CudaMemory.getGpuStats(device).usedMemory.addAndGet(length);
            return new ReferenceWrapper<>(alloc, x -> {
              MemoryType.logger.debug(RefString.format("Freed %s %s (%s bytes) in device %s via %s", name(),
                  Integer.toHexString(RefSystem.identityHashCode(alloc)), length, device,
                  CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : ""));
              CudaMemory.getGpuStats(device).usedMemory.addAndGet(-length);
              MemoryType.this.free(x, device);
            });
          });
        }

        @Override
        public void reset(final @RefAware ReferenceWrapper<CudaPointer> data,
                          final long size) {
          RefUtil.freeRef(data);
          // There is no need to clean new objects - native memory system doesn't either.
        }

        @Override
        protected void free(@Nonnull final @RefAware ReferenceWrapper<CudaPointer> obj) {
          MemoryType.logger.debug(RefString.format("Freed %s %s in device %s at %s", name(),
              Integer.toHexString(RefSystem.identityHashCode(obj.peek())), device,
              CudaSettings.INSTANCE().profileMemoryIO ? Util.getCaller() : ""));
          obj.destroy();
          RefUtil.freeRef(obj);
        }
      }.setPersistanceMode(CudaSettings.INSTANCE().memoryCacheMode).setMinLengthPerBuffer(1).setMaxItemsPerBuffer(10)
          .setPurgeFreq(CudaSettings.INSTANCE().memoryCacheTTL);
    });
    temp_76_0002.freeRef();
    return temp_76_0001;
  }

}
