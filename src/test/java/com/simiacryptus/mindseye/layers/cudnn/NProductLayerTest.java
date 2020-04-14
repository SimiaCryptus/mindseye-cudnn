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
 * The type N product layer test.
 */
public abstract class NProductLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;

  /**
   * Instantiates a new N product layer test.
   *
   * @param precision the precision
   */
  public NProductLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}, {1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    NProductLayer temp_54_0002 = new NProductLayer();
    temp_54_0002.setPrecision(precision);
    NProductLayer temp_54_0001 = RefUtil.addRef(temp_54_0002);
    temp_54_0002.freeRef();
    return temp_54_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}, {8, 8, 1}};
  }

  /**
   * The type Double.
   */
  public static class Double extends NProductLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }

  }

  /**
   * The type Double 3.
   */
  public static class Double3 extends NProductLayerTest {
    /**
     * Instantiates a new Double 3.
     */
    public Double3() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{1200, 1200, 3}, {1200, 1200, 3}, {1200, 1200, 3}};
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{8, 8, 1}, {8, 8, 1}, {8, 8, 1}};
    }
  }

  /**
   * The type Float.
   */
  public static class Float extends NProductLayerTest {
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
