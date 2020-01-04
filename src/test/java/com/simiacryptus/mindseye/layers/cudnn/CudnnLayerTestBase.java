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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.notebook.NotebookOutput;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;

public abstract @com.simiacryptus.ref.lang.RefAware class CudnnLayerTestBase extends LayerTestBase {

  @Override
  public void run(@NotNull @Nonnull NotebookOutput log) {
    OutputStream file = log.file("gpu.log");
    log.out(log.link(new File(log.getResourceDir(), "gpu.log"), "Cuda Log"));
    CudaSystem.addLog(new PrintStream(file));
    super.run(log);
  }

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") CudnnLayerTestBase addRef() {
    return (CudnnLayerTestBase) super.addRef();
  }

  public static @SuppressWarnings("unused") CudnnLayerTestBase[] addRefs(CudnnLayerTestBase[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CudnnLayerTestBase::addRef)
        .toArray((x) -> new CudnnLayerTestBase[x]);
  }

  public static @SuppressWarnings("unused") CudnnLayerTestBase[][] addRefs(CudnnLayerTestBase[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CudnnLayerTestBase::addRefs)
        .toArray((x) -> new CudnnLayerTestBase[x][]);
  }
}
