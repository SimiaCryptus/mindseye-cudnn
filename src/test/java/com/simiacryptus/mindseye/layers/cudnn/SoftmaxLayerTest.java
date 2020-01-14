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
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class SoftmaxLayerTest extends CudnnLayerTestBase {

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

  @Nullable
  public static @SuppressWarnings("unused")
  SoftmaxLayerTest[] addRefs(@Nullable SoftmaxLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayerTest::addRef)
        .toArray((x) -> new SoftmaxLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SoftmaxLayerTest[][] addRefs(@Nullable SoftmaxLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayerTest::addRefs)
        .toArray((x) -> new SoftmaxLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    SoftmaxActivationLayer temp_70_0002 = new SoftmaxActivationLayer();
    SoftmaxActivationLayer temp_70_0003 = temp_70_0002.setMode(mode);
    SoftmaxActivationLayer temp_70_0001 = temp_70_0003.setAlgorithm(algorithm);
    temp_70_0003.freeRef();
    temp_70_0002.freeRef();
    return temp_70_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SoftmaxLayerTest addRef() {
    return (SoftmaxLayerTest) super.addRef();
  }

  public static class Basic extends SoftmaxLayerTest {
    public Basic() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.INSTANCE);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Basic[] addRefs(@Nullable Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

  public static class Pixel extends SoftmaxLayerTest {
    public Pixel() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Pixel[] addRefs(@Nullable Pixel[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Pixel::addRef).toArray((x) -> new Pixel[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Pixel addRef() {
      return (Pixel) super.addRef();
    }
  }

  public static class PixelLog extends SoftmaxLayerTest {
    public PixelLog() {
      super(SoftmaxActivationLayer.SoftmaxAlgorithm.LOG, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    PixelLog[] addRefs(@Nullable PixelLog[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(PixelLog::addRef).toArray((x) -> new PixelLog[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    PixelLog addRef() {
      return (PixelLog) super.addRef();
    }
  }
}
