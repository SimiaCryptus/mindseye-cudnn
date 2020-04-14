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

/**
 * The type Img tile cycle layer test.
 */
public abstract class ImgTileCycleLayerTest extends CudnnLayerTestBase {

  /**
   * Instantiates a new Img tile cycle layer test.
   */
  public ImgTileCycleLayerTest() {
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new ImgTileCycleLayer();
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type One third.
   */
  public static class OneThird extends ImgTileCycleLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgTileCycleLayer temp_71_0002 = new ImgTileCycleLayer();
      temp_71_0002.setXPos(0.3);
      ImgTileCycleLayer temp_71_0003 = temp_71_0002.addRef();
      temp_71_0003.setYPos(0.3);
      ImgTileCycleLayer temp_71_0001 = temp_71_0003.addRef();
      temp_71_0003.freeRef();
      temp_71_0002.freeRef();
      return temp_71_0001;
    }

  }

  /**
   * The type Basic.
   */
  public static class Basic extends ImgTileCycleLayerTest {

  }
}
