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
import com.simiacryptus.mindseye.layers.cudnn.ImgCropLayer.Alignment;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class ImgPaddingLayerTest extends CudnnLayerTestBase {

  private static final int SIZE_OUT = 4;
  private static final int SIZE_IN = 2;

  public ImgPaddingLayerTest() {
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgPaddingLayerTest[] addRefs(@Nullable ImgPaddingLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPaddingLayerTest::addRef)
        .toArray((x) -> new ImgPaddingLayerTest[x]);
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  @Nonnull
  @Override
  public abstract Layer getLayer(final int[][] inputSize, Random random);

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPaddingLayerTest addRef() {
    return (ImgPaddingLayerTest) super.addRef();
  }

  public static class Center extends ImgPaddingLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{SIZE_IN, SIZE_IN, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgPaddingLayer(SIZE_OUT, SIZE_OUT);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Center addRef() {
      return (Center) super.addRef();
    }
  }

  public static class Left extends ImgPaddingLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{SIZE_IN, SIZE_IN, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgPaddingLayer temp_63_0002 = new ImgPaddingLayer(SIZE_OUT, SIZE_OUT);
      temp_63_0002.setHorizontalAlign(Alignment.Left);
      ImgPaddingLayer temp_63_0001 = temp_63_0002.addRef();
      temp_63_0002.freeRef();
      return temp_63_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Left addRef() {
      return (Left) super.addRef();
    }
  }

  public static class Right extends ImgPaddingLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{SIZE_IN, SIZE_IN, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgPaddingLayer temp_63_0004 = new ImgPaddingLayer(SIZE_OUT, SIZE_OUT);
      temp_63_0004.setHorizontalAlign(Alignment.Right);
      ImgPaddingLayer temp_63_0003 = temp_63_0004.addRef();
      temp_63_0004.freeRef();
      return temp_63_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Right addRef() {
      return (Right) super.addRef();
    }
  }

  public static class Top extends ImgPaddingLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{SIZE_IN, SIZE_IN, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgPaddingLayer temp_63_0006 = new ImgPaddingLayer(SIZE_OUT, SIZE_OUT);
      temp_63_0006.setVerticalAlign(Alignment.Left);
      ImgPaddingLayer temp_63_0005 = temp_63_0006.addRef();
      temp_63_0006.freeRef();
      return temp_63_0005;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Top addRef() {
      return (Top) super.addRef();
    }
  }

  public static class Bottom extends ImgPaddingLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{SIZE_IN, SIZE_IN, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgPaddingLayer temp_63_0008 = new ImgPaddingLayer(SIZE_OUT, SIZE_OUT);
      temp_63_0008.setVerticalAlign(Alignment.Right);
      ImgPaddingLayer temp_63_0007 = temp_63_0008.addRef();
      temp_63_0008.freeRef();
      return temp_63_0007;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Bottom addRef() {
      return (Bottom) super.addRef();
    }
  }

}
