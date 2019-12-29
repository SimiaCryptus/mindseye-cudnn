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

import javax.annotation.Nonnull;
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

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  @Nonnull
  @Override
  public abstract Layer getLayer(final int[][] inputSize, Random random);

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
      return new ImgPaddingLayer(SIZE_OUT, SIZE_OUT).setHorizontalAlign(Alignment.Left);
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
      return new ImgPaddingLayer(SIZE_OUT, SIZE_OUT).setHorizontalAlign(Alignment.Right);
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
      return new ImgPaddingLayer(SIZE_OUT, SIZE_OUT).setVerticalAlign(Alignment.Left);
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
      return new ImgPaddingLayer(SIZE_OUT, SIZE_OUT).setVerticalAlign(Alignment.Right);
    }

  }

}
