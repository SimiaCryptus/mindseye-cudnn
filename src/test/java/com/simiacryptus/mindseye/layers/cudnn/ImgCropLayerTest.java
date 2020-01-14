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

public abstract class ImgCropLayerTest extends CudnnLayerTestBase {

  public ImgCropLayerTest() {
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgCropLayerTest[] addRefs(@Nullable ImgCropLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayerTest::addRef)
        .toArray((x) -> new ImgCropLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgCropLayerTest[][] addRefs(@Nullable ImgCropLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayerTest::addRefs)
        .toArray((x) -> new ImgCropLayerTest[x][]);
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
  ImgCropLayerTest addRef() {
    return (ImgCropLayerTest) super.addRef();
  }

  public static class Center extends ImgCropLayerTest {
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Center[] addRefs(@Nullable Center[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Center::addRef).toArray((x) -> new Center[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2);
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

  public static class Left extends ImgCropLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Left[] addRefs(@Nullable Left[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Left::addRef).toArray((x) -> new Left[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgCropLayer temp_57_0002 = new ImgCropLayer(2, 2);
      ImgCropLayer temp_57_0001 = temp_57_0002.setHorizontalAlign(ImgCropLayer.Alignment.Left);
      temp_57_0002.freeRef();
      return temp_57_0001;
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

  public static class Right extends ImgCropLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Right[] addRefs(@Nullable Right[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Right::addRef).toArray((x) -> new Right[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgCropLayer temp_57_0004 = new ImgCropLayer(2, 2);
      ImgCropLayer temp_57_0003 = temp_57_0004.setHorizontalAlign(ImgCropLayer.Alignment.Right);
      temp_57_0004.freeRef();
      return temp_57_0003;
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

  public static class Top extends ImgCropLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Top[] addRefs(@Nullable Top[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Top::addRef).toArray((x) -> new Top[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgCropLayer temp_57_0006 = new ImgCropLayer(2, 2);
      ImgCropLayer temp_57_0005 = temp_57_0006.setVerticalAlign(ImgCropLayer.Alignment.Left);
      temp_57_0006.freeRef();
      return temp_57_0005;
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

  public static class Bottom extends ImgCropLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Bottom[] addRefs(@Nullable Bottom[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Bottom::addRef).toArray((x) -> new Bottom[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgCropLayer temp_57_0008 = new ImgCropLayer(2, 2);
      ImgCropLayer temp_57_0007 = temp_57_0008.setVerticalAlign(ImgCropLayer.Alignment.Left);
      temp_57_0008.freeRef();
      return temp_57_0007;
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
