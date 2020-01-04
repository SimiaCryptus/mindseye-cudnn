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
class ImgTileCycleLayerTest extends CudnnLayerTestBase {

  public ImgTileCycleLayerTest() {
    validateBatchExecution = false;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  public static @SuppressWarnings("unused")
  ImgTileCycleLayerTest[] addRefs(ImgTileCycleLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayerTest::addRef)
        .toArray((x) -> new ImgTileCycleLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileCycleLayerTest[][] addRefs(ImgTileCycleLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayerTest::addRefs)
        .toArray((x) -> new ImgTileCycleLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgTileCycleLayer();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgTileCycleLayerTest addRef() {
    return (ImgTileCycleLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class OneThird extends ImgTileCycleLayerTest {

    public static @SuppressWarnings("unused")
    OneThird[] addRefs(OneThird[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(OneThird::addRef)
          .toArray((x) -> new OneThird[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgTileCycleLayer().setXPos(0.3).setYPos(0.3);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    OneThird addRef() {
      return (OneThird) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends ImgTileCycleLayerTest {

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
}
