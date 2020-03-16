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
import org.junit.jupiter.api.TestInfo;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract class ImgZeroPaddingLayerTest extends CudnnLayerTestBase {

  public ImgZeroPaddingLayerTest() {
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgZeroPaddingLayer.class;
  }

  @Override
  @Disabled
  public void batchingTest(TestInfo testInfo) {
    super.batchingTest(testInfo);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgZeroPaddingLayer(1, 1);
  }

  public static class Basic extends ImgZeroPaddingLayerTest {

  }

  //  /**
  //   * Basic Test
  //   */
  //  public static class Asymmetric extends ImgZeroPaddingLayerTest {
  //
  //    @Override
  //    public int[][] getDims() {
  //      return new int[][]{
  //        {5, 3, 2}
  //      };
  //    }
  //
  //    @Override
  //    public int[][] getLargeDims() {
  //      return new int[][]{
  //        {5, 5, 3}
  //      };
  //    }
  //  }

}
