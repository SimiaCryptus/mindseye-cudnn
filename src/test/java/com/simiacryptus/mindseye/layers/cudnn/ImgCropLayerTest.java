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

/**
 * The type Img crop layer test.
 */
public abstract class ImgCropLayerTest extends CudnnLayerTestBase {

  /**
   * Instantiates a new Img crop layer test.
   */
  public ImgCropLayerTest() {
  }

  @Nonnull
  @Override
  public abstract Layer getLayer();

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims();

  /**
   * The type Center.
   */
  public static class Center extends ImgCropLayerTest {
    @Nonnull
    @Override
    public Layer getLayer() {
      return new ImgCropLayer(2, 2);
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

  }

  /**
   * The type Left.
   */
  public static class Left extends ImgCropLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgCropLayer temp_57_0002 = new ImgCropLayer(2, 2);
      temp_57_0002.setHorizontalAlign(ImgCropLayer.Alignment.Left);
      ImgCropLayer temp_57_0001 = temp_57_0002.addRef();
      temp_57_0002.freeRef();
      return temp_57_0001;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

  }

  /**
   * The type Right.
   */
  public static class Right extends ImgCropLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgCropLayer temp_57_0004 = new ImgCropLayer(2, 2);
      temp_57_0004.setHorizontalAlign(ImgCropLayer.Alignment.Right);
      ImgCropLayer temp_57_0003 = temp_57_0004.addRef();
      temp_57_0004.freeRef();
      return temp_57_0003;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

  }

  /**
   * The type Top.
   */
  public static class Top extends ImgCropLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgCropLayer temp_57_0006 = new ImgCropLayer(2, 2);
      temp_57_0006.setVerticalAlign(ImgCropLayer.Alignment.Left);
      ImgCropLayer temp_57_0005 = temp_57_0006.addRef();
      temp_57_0006.freeRef();
      return temp_57_0005;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

  }

  /**
   * The type Bottom.
   */
  public static class Bottom extends ImgCropLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgCropLayer temp_57_0008 = new ImgCropLayer(2, 2);
      temp_57_0008.setVerticalAlign(ImgCropLayer.Alignment.Left);
      ImgCropLayer temp_57_0007 = temp_57_0008.addRef();
      temp_57_0008.freeRef();
      return temp_57_0007;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }
  }

}
