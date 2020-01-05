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
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import org.jetbrains.annotations.NotNull;
import org.junit.Ignore;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class RescaledSubnetLayerTest extends CudnnLayerTestBase {

  @Nonnull
  final ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 1, 1);

  public RescaledSubnetLayerTest() {

  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer.class;
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayerTest[] addRefs(RescaledSubnetLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayerTest::addRef)
        .toArray((x) -> new RescaledSubnetLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayerTest[][] addRefs(RescaledSubnetLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayerTest::addRefs)
        .toArray((x) -> new RescaledSubnetLayerTest[x][]);
  }

  @Override
  public void run(@NotNull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    //    CudaSystem.addLog(apiLog);
    super.run(log);
    //    apiLog.close();
    //    CudaSystem.apiLog.remove(apiLog);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new RescaledSubnetLayer(2, convolutionLayer.set(() -> this.random()));
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  RescaledSubnetLayerTest addRef() {
    return (RescaledSubnetLayerTest) super.addRef();
  }

  public static @RefAware
  class Basic extends RescaledSubnetLayerTest {
    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    @Override
    @Test(timeout = 15 * 60 * 1000)
    @Ignore // Crashing SpanBug!?!?
    public void test() throws Throwable {
      super.test();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
