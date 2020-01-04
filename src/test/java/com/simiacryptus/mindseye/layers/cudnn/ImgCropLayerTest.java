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

public abstract @com.simiacryptus.ref.lang.RefAware class ImgCropLayerTest extends CudnnLayerTestBase {

  public ImgCropLayerTest() {
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  @Nonnull
  @Override
  public abstract Layer getLayer(final int[][] inputSize, Random random);

  public static @com.simiacryptus.ref.lang.RefAware class Center extends ImgCropLayerTest {
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 4, 4, 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Center addRef() {
      return (Center) super.addRef();
    }

    public static @SuppressWarnings("unused") Center[] addRefs(Center[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Center::addRef).toArray((x) -> new Center[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Left extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 4, 4, 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setHorizontalAlign(ImgCropLayer.Alignment.Left);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Left addRef() {
      return (Left) super.addRef();
    }

    public static @SuppressWarnings("unused") Left[] addRefs(Left[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Left::addRef).toArray((x) -> new Left[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Right extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 4, 4, 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setHorizontalAlign(ImgCropLayer.Alignment.Right);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Right addRef() {
      return (Right) super.addRef();
    }

    public static @SuppressWarnings("unused") Right[] addRefs(Right[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Right::addRef).toArray((x) -> new Right[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Top extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 4, 4, 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setVerticalAlign(ImgCropLayer.Alignment.Left);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Top addRef() {
      return (Top) super.addRef();
    }

    public static @SuppressWarnings("unused") Top[] addRefs(Top[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Top::addRef).toArray((x) -> new Top[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Bottom extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 4, 4, 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setVerticalAlign(ImgCropLayer.Alignment.Left);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Bottom addRef() {
      return (Bottom) super.addRef();
    }

    public static @SuppressWarnings("unused") Bottom[] addRefs(Bottom[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Bottom::addRef).toArray((x) -> new Bottom[x]);
    }

  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgCropLayerTest addRef() {
    return (ImgCropLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgCropLayerTest[] addRefs(ImgCropLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayerTest::addRef)
        .toArray((x) -> new ImgCropLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ImgCropLayerTest[][] addRefs(ImgCropLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayerTest::addRefs)
        .toArray((x) -> new ImgCropLayerTest[x][]);
  }

}
