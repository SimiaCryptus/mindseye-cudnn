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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.function.ToIntFunction;

/**
 * The type Cuda resource.
 *
 * @param <T> the type parameter
 */
public class CudaResource<T> extends CudaResourceBase<T> {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaResource.class);

  /**
   * The Device id.
   */
  public final int deviceId;
  private final ToIntFunction<T> destructor;

  /**
   * Instantiates a new Cuda resource.
   *
   * @param obj        the obj
   * @param destructor the destructor
   * @param deviceId   the device id
   */
  protected CudaResource(final T obj, final ToIntFunction<T> destructor, int deviceId) {
    super(obj);
    this.destructor = destructor;
    this.deviceId = deviceId;
  }

  @Override
  public int getDeviceId() {
    return deviceId;
  }

  public void release() {
    try {
      if (isActiveObj()) {
        CudaSystem.withDevice(deviceId, dev -> {
          CudaSystem.handle(this.destructor.applyAsInt(ptr));
          dev.freeRef();
        });
      }
    } catch (@Nonnull final Throwable e) {
      CudaResource.logger.debug("Error freeing resource " + this, e);
    }
  }

  public void _free() {
    super._free();
    cleanup();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CudaResource<T> addRef() {
    return (CudaResource<T>) super.addRef();
  }
}
