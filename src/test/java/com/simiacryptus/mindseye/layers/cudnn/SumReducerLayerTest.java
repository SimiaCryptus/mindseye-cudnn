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
 * The type Sum reducer layer test.
 */
public abstract class SumReducerLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;

  /**
   * Instantiates a new Sum reducer layer test.
   *
   * @param precision the precision
   */
  public SumReducerLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    SumReducerLayer temp_59_0002 = new SumReducerLayer();
    temp_59_0002.setPrecision(precision);
    SumReducerLayer temp_59_0001 = RefUtil.addRef(temp_59_0002);
    temp_59_0002.freeRef();
    return temp_59_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}};
  }

  /**
   * The type Double.
   */
  public static class Double extends SumReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }

  }

  /**
   * The type Asymmetric.
   */
  public static class Asymmetric extends SumReducerLayerTest {
    /**
     * Instantiates a new Asymmetric.
     */
    public Asymmetric() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{1000, 600, 3}};
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
  public static class Float extends SumReducerLayerTest {
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
