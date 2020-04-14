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

/**
 * The type Mean sq loss layer test.
 */
public abstract class MeanSqLossLayerTest extends CudnnLayerTestBase {

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}, {1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new MeanSqLossLayer();
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.MeanSqLossLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}, {8, 8, 1}};
  }

  /**
   * The type Basic.
   */
  public static class Basic extends MeanSqLossLayerTest {
  }

  /**
   * The type Asymetric.
   */
  public static class Asymetric extends MeanSqLossLayerTest {

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{200, 300, 100}, {200, 300, 100}};
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{2, 3, 1}, {2, 3, 1}};
    }
  }

}
