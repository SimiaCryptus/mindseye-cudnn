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
import jcuda.runtime.cudaStream_t;

import java.util.Arrays;

public @RefAware
class CudaStream extends CudaResource<cudaStream_t> {
  CudaStream(cudaStream_t stream) {
    super(stream, CudaSystem::cudaStreamDestroy, CudaSystem.getThreadDeviceId());
  }

  public static @SuppressWarnings("unused")
  CudaStream[] addRefs(CudaStream[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaStream::addRef)
        .toArray((x) -> new CudaStream[x]);
  }

  public static @SuppressWarnings("unused")
  CudaStream[][] addRefs(CudaStream[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CudaStream::addRefs)
        .toArray((x) -> new CudaStream[x][]);
  }

  public void sync() {
    CudaSystem.cudaStreamSynchronize(getPtr());
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  CudaStream addRef() {
    return (CudaStream) super.addRef();
  }
}
