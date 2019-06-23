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

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class CudaResourceBase<T> extends ReferenceCountingBase implements CudaSystem.CudaDeviceResource {
  private static final Logger logger = LoggerFactory.getLogger(CudaResourceBase.class);
  public final int objGeneration = CudaSystem.gpuGeneration.get();
  protected T ptr;

  public CudaResourceBase(final T obj) {
    this.ptr = obj;
  }

  public T getPtr() {
    assertAlive();
    return ptr;
  }

  protected abstract void _free();

  public boolean isActiveObj() {
    return objGeneration == CudaSystem.gpuGeneration.get();
  }

  public abstract void release();
}
