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
 * The type Band reducer layer test.
 */
public abstract class BandReducerLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;
  private final double alpha;

  /**
   * Instantiates a new Band reducer layer test.
   *
   * @param precision the precision
   * @param alpha     the alpha
   */
  public BandReducerLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{32, 32, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    BandReducerLayer temp_66_0002 = new BandReducerLayer();
    temp_66_0002.setAlpha(alpha);
    BandReducerLayer temp_66_0003 = temp_66_0002.addRef();
    temp_66_0003.setPrecision(precision);
    BandReducerLayer temp_66_0001 = RefUtil.addRef(temp_66_0003);
    temp_66_0003.freeRef();
    temp_66_0002.freeRef();
    return temp_66_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2, 1}};
  }

  /**
   * The type Double.
   */
  public static class Double extends BandReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 1.0);
    }
  }

  /**
   * The type Negative.
   */
  public static class Negative extends BandReducerLayerTest {
    /**
     * Instantiates a new Negative.
     */
    public Negative() {
      super(Precision.Double, -5.0);
    }
  }

  /**
   * The type Asymmetric.
   */
  public static class Asymmetric extends BandReducerLayerTest {
    /**
     * Instantiates a new Asymmetric.
     */
    public Asymmetric() {
      super(Precision.Double, 1.0);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{200, 100, 3}};
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3, 5, 2}};
    }
  }

  /**
   * The type Float.
   */
  public static class Float extends BandReducerLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 1.0);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }
}
