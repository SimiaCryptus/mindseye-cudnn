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

import com.simiacryptus.ref.lang.ReferenceCountingBase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Cuda resource base.
 *
 * @param <T> the type parameter
 */
public abstract class CudaResourceBase<T> extends ReferenceCountingBase implements CudaSystem.CudaDeviceResource {
  private static final Logger logger = LoggerFactory.getLogger(CudaResourceBase.class);
  /**
   * The Obj generation.
   */
  public final int objGeneration = CudaSystem.gpuGeneration.get();
  /**
   * The Ptr.
   */
  @Nullable
  protected T ptr;

  /**
   * Instantiates a new Cuda resource base.
   *
   * @param obj the obj
   */
  public CudaResourceBase(final T obj) {
    this.ptr = obj;
  }

  /**
   * Gets ptr.
   *
   * @return the ptr
   */
  @Nullable
  public T getPtr() {
    assertAlive();
    return ptr;
  }

  /**
   * Is active obj boolean.
   *
   * @return the boolean
   */
  public boolean isActiveObj() {
    return objGeneration == CudaSystem.gpuGeneration.get();
  }

  /**
   * Release.
   */
  public abstract void release();

  public void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CudaResourceBase<T> addRef() {
    return (CudaResourceBase<T>) super.addRef();
  }

  /**
   * Cleanup.
   */
  protected void cleanup() {
    CudnnHandle threadHandle = CudnnHandle.threadContext.get();
    if (null != threadHandle) {
      threadHandle.cleanupNative.add(this);
      threadHandle.freeRef();
    } else
      release();
  }
}
