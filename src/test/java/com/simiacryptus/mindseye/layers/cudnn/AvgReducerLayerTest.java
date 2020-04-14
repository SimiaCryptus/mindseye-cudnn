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

import javax.annotation.Nonnull;

/**
 * The type Avg reducer layer test.
 */
public abstract class AvgReducerLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Avg reducer layer test.
   *
   * @param precision the precision
   * @param smallSize the small size
   * @param largeSize the large size
   */
  public AvgReducerLayerTest(final Precision precision, final int smallSize, final int largeSize) {
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    AvgReducerLayer temp_61_0002 = new AvgReducerLayer();
    temp_61_0002.setPrecision(precision);
    AvgReducerLayer temp_61_0001 = RefUtil.addRef(temp_61_0002);
    temp_61_0002.freeRef();
    return temp_61_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, 1}};
  }

  /**
   * The type Double.
   */
  public static class Double extends AvgReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 2, 1200);
    }

  }

  /**
   * The type Asymmetric.
   */
  public static class Asymmetric extends AvgReducerLayerTest {
    /**
     * Instantiates a new Asymmetric.
     */
    public Asymmetric() {
      super(Precision.Double, 2, 1200);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{1200, 800, 3}};
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{2, 5, 2}};
    }

  }

  /**
   * The type Float.
   */
  public static class Float extends AvgReducerLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 2, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
