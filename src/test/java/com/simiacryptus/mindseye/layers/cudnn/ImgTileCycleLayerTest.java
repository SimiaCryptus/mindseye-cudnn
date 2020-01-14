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

public abstract class ImgTileCycleLayerTest extends CudnnLayerTestBase {

  public ImgTileCycleLayerTest() {
    validateBatchExecution = false;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileCycleLayerTest[] addRefs(@Nullable ImgTileCycleLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayerTest::addRef)
        .toArray((x) -> new ImgTileCycleLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileCycleLayerTest[][] addRefs(@Nullable ImgTileCycleLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileCycleLayerTest::addRefs)
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileCycleLayerTest addRef() {
    return (ImgTileCycleLayerTest) super.addRef();
  }

  public static class OneThird extends ImgTileCycleLayerTest {

    @Nullable
    public static @SuppressWarnings("unused")
    OneThird[] addRefs(@Nullable OneThird[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(OneThird::addRef).toArray((x) -> new OneThird[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgTileCycleLayer temp_71_0002 = new ImgTileCycleLayer();
      ImgTileCycleLayer temp_71_0003 = temp_71_0002.setXPos(0.3);
      ImgTileCycleLayer temp_71_0001 = temp_71_0003.setYPos(0.3);
      temp_71_0003.freeRef();
      temp_71_0002.freeRef();
      return temp_71_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    OneThird addRef() {
      return (OneThird) super.addRef();
    }

  }

  public static class Basic extends ImgTileCycleLayerTest {

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
}
