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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefUtil;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;

/**
 * The type Img band dynamic bias layer test.
 */
public abstract class ImgBandDynamicBiasLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;

  /**
   * Instantiates a new Img band dynamic bias layer test.
   *
   * @param precision the precision
   */
  public ImgBandDynamicBiasLayerTest(final Precision precision) {
    this.precision = precision;
    this.testingBatchSize = 1;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}, {1, 1, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    ImgBandDynamicBiasLayer temp_68_0004 = new ImgBandDynamicBiasLayer();
    temp_68_0004.setPrecision(precision);
    ImgBandDynamicBiasLayer temp_68_0003 = RefUtil.addRef(temp_68_0004);
    temp_68_0004.freeRef();
    return temp_68_0003;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 3}, {1, 1, 3}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  @Override
  protected @Nonnull Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  /**
   * The type Double.
   */
  public static class Double extends ImgBandDynamicBiasLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }

  /**
   * The type Float.
   */
  public static class Float extends ImgBandDynamicBiasLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
