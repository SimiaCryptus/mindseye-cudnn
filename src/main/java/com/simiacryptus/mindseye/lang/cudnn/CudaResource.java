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

import com.simiacryptus.ref.lang.RefAware;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.function.ToIntFunction;

public @RefAware
class CudaResource<T> extends CudaResourceBase<T> {
  protected static final Logger logger = LoggerFactory.getLogger(CudaResource.class);

  public final int deviceId;
  private final ToIntFunction<T> destructor;

  protected CudaResource(final T obj, final ToIntFunction<T> destructor, int deviceId) {
    super(obj);
    this.destructor = destructor;
    this.deviceId = deviceId;
  }

  @Override
  public int getDeviceId() {
    return deviceId;
  }

  public static @SuppressWarnings("unused")
  CudaResource[] addRefs(CudaResource[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaResource::addRef)
        .toArray((x) -> new CudaResource[x]);
  }

  public static @SuppressWarnings("unused")
  CudaResource[][] addRefs(CudaResource[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaResource::addRefs)
        .toArray((x) -> new CudaResource[x][]);
  }

  public void release() {
    try {
      if (isActiveObj()) {
        CudaSystem.withDevice(deviceId, dev -> {
          CudaSystem.handle(this.destructor.applyAsInt(ptr));
        });
      }
    } catch (@Nonnull final Throwable e) {
      CudaResource.logger.debug("Error freeing resource " + this, e);
    }
  }

  public void _free() {
    CudnnHandle threadHandle = CudaSystem.getThreadHandle();
    if (null != threadHandle)
      threadHandle.cleanupNative.add(this);
    else
      release();
  }

  public @Override
  @SuppressWarnings("unused")
  CudaResource<T> addRef() {
    return (CudaResource<T>) super.addRef();
  }
}
