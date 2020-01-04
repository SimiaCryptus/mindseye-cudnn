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

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class SoftmaxLayerTest extends CudnnLayerTestBase {

  private final SoftmaxActivationLayer.SoftmaxAlgorithm algorithm;
  private final SoftmaxActivationLayer.SoftmaxMode mode;

  public SoftmaxLayerTest(final SoftmaxActivationLayer.SoftmaxAlgorithm algorithm,
                          final SoftmaxActivationLayer.SoftmaxMode mode) {
    this.algorithm = algorithm;
    this.mode = mode;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
    //return com.simiacryptus.mindseye.layers.java.SoftmaxLayer.class;
  }

  public static @SuppressWarnings("unused")
  SoftmaxLayerTest[] addRefs(SoftmaxLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayerTest::addRef)
        .toArray((x) -> new SoftmaxLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  SoftmaxLayerTest[][] addRefs(SoftmaxLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayerTest::addRefs)
        .toArray((x) -> new SoftmaxLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new SoftmaxActivationLayer().setMode(mode).setAlgorithm(algorithm);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SoftmaxLayerTest addRef() {
    return (SoftmaxLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends SoftmaxLayerTest {
    public Basic() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.INSTANCE);
    }

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
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

  public static @com.simiacryptus.ref.lang.RefAware
  class Pixel extends SoftmaxLayerTest {
    public Pixel() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    }

    public static @SuppressWarnings("unused")
    Pixel[] addRefs(Pixel[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Pixel::addRef).toArray((x) -> new Pixel[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Pixel addRef() {
      return (Pixel) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class PixelLog extends SoftmaxLayerTest {
    public PixelLog() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.LOG, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    }

    public static @SuppressWarnings("unused")
    PixelLog[] addRefs(PixelLog[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(PixelLog::addRef)
          .toArray((x) -> new PixelLog[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    PixelLog addRef() {
      return (PixelLog) super.addRef();
    }
  }
}
