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


public abstract class ImgCropLayerTest extends CudnnLayerTestBase {

  public ImgCropLayerTest() {
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  @Nonnull
  @Override
  public abstract Layer getLayer(final int[][] inputSize, Random random);

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  public static class Center extends ImgCropLayerTest {
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

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
    }

  }

  public static class Left extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setHorizontalAlign(ImgCropLayer.Alignment.Left);
    }

  }

  public static class Right extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setHorizontalAlign(ImgCropLayer.Alignment.Right);
    }

  }

  public static class Top extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setVerticalAlign(ImgCropLayer.Alignment.Left);
    }

  }

  public static class Bottom extends ImgCropLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgCropLayer(2, 2).setVerticalAlign(ImgCropLayer.Alignment.Left);
    }

  }

}
