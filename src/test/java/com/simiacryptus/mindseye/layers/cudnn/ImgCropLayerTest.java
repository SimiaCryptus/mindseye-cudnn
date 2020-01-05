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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class ImgCropLayerTest extends CudnnLayerTestBase {

  public ImgCropLayerTest() {
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  public static @SuppressWarnings("unused")
  ImgCropLayerTest[] addRefs(ImgCropLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgCropLayerTest::addRef)
        .toArray((x) -> new ImgCropLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgCropLayerTest[][] addRefs(ImgCropLayerTest[][] array) {
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

  public @Override
  @SuppressWarnings("unused")
  ImgCropLayerTest addRef() {
    return (ImgCropLayerTest) super.addRef();
  }

  public static @RefAware
  class Center extends ImgCropLayerTest {
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
    }

    public static @SuppressWarnings("unused")
    Center[] addRefs(Center[] array) {
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

    public @Override
    @SuppressWarnings("unused")
    Center addRef() {
      return (Center) super.addRef();
    }

  }

  public static @RefAware
  class Left extends ImgCropLayerTest {
    public static @SuppressWarnings("unused")
    Left[] addRefs(Left[] array) {
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
      ImgCropLayer temp_57_0001 = temp_57_0002
          .setHorizontalAlign(ImgCropLayer.Alignment.Left);
      if (null != temp_57_0002)
        temp_57_0002.freeRef();
      return temp_57_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Left addRef() {
      return (Left) super.addRef();
    }

  }

  public static @RefAware
  class Right extends ImgCropLayerTest {
    public static @SuppressWarnings("unused")
    Right[] addRefs(Right[] array) {
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
      ImgCropLayer temp_57_0003 = temp_57_0004
          .setHorizontalAlign(ImgCropLayer.Alignment.Right);
      if (null != temp_57_0004)
        temp_57_0004.freeRef();
      return temp_57_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Right addRef() {
      return (Right) super.addRef();
    }

  }

  public static @RefAware
  class Top extends ImgCropLayerTest {
    public static @SuppressWarnings("unused")
    Top[] addRefs(Top[] array) {
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
      ImgCropLayer temp_57_0005 = temp_57_0006
          .setVerticalAlign(ImgCropLayer.Alignment.Left);
      if (null != temp_57_0006)
        temp_57_0006.freeRef();
      return temp_57_0005;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Top addRef() {
      return (Top) super.addRef();
    }

  }

  public static @RefAware
  class Bottom extends ImgCropLayerTest {
    public static @SuppressWarnings("unused")
    Bottom[] addRefs(Bottom[] array) {
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
      ImgCropLayer temp_57_0007 = temp_57_0008
          .setVerticalAlign(ImgCropLayer.Alignment.Left);
      if (null != temp_57_0008)
        temp_57_0008.freeRef();
      return temp_57_0007;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Bottom addRef() {
      return (Bottom) super.addRef();
    }

  }

}
