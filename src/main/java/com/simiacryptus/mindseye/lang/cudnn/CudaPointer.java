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

import jcuda.Pointer;

import javax.annotation.Nonnull;

/**
 * The type Cuda pointer.
 */
public class CudaPointer extends Pointer {

  /**
   * The Device id.
   */
  public final int deviceId = CudaSystem.getThreadDeviceId();

  /**
   * Instantiates a new Cuda pointer.
   *
   * @param other the other
   */
  public CudaPointer(@Nonnull final Pointer other) {
    super(other);
  }

  /**
   * Instantiates a new Cuda pointer.
   *
   * @param other      the other
   * @param byteOffset the byte offset
   */
  public CudaPointer(@Nonnull final Pointer other, final long byteOffset) {
    super(other, byteOffset);
  }

  /**
   * Instantiates a new Cuda pointer.
   */
  public CudaPointer() {
    super();
  }

  @Override
  public long getByteOffset() {
    return super.getByteOffset();
  }

  /**
   * To cuda pointer.
   *
   * @param values the values
   * @return the cuda pointer
   */
  @Nonnull
  public static CudaPointer to(@Nonnull float values[]) {
    return new CudaPointer(Pointer.to(values));
  }

  /**
   * To cuda pointer.
   *
   * @param values the values
   * @return the cuda pointer
   */
  @Nonnull
  public static CudaPointer to(@Nonnull double values[]) {
    return new CudaPointer(Pointer.to(values));
  }

  @Nonnull
  @Override
  public CudaPointer withByteOffset(final long byteOffset) {
    return new CudaPointer(this, byteOffset);
  }

}
