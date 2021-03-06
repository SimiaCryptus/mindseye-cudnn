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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Img tile assembly layer test.
 */
public abstract class ImgTileAssemblyLayerTest extends CudnnLayerTestBase {

  /**
   * Instantiates a new Img tile assembly layer test.
   */
  public ImgTileAssemblyLayerTest() {
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    int scale = 25;
    return new int[][]{
        {2 * scale, 2 * scale, scale},
        {scale, 2 * scale, scale},
        {2 * scale, 2 * scale, scale},
        {scale, 2 * scale, scale},
        {2 * scale, scale, scale},
        {scale, scale, scale}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new ImgTileAssemblyLayer(2, 3);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2, 1}, {1, 2, 1}, {2, 2, 1}, {1, 2, 1}, {2, 1, 1}, {1, 1, 1}
        //      {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}, {3, 3, 1}
    };
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends ImgTileAssemblyLayerTest {

  }

}
