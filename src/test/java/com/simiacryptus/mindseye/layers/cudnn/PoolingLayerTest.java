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

import javax.annotation.Nonnull;

public abstract class PoolingLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public PoolingLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{800, 800, 16}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setPrecision(precision);
    return poolingLayer;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}};
  }

  public static class Repro extends PoolingLayerTest {
    public Repro() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{3, 2, 512}};
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setWindowXY(3, 2);
      poolingLayer.setStrideXY(3, 2);
      poolingLayer.setPrecision(precision);
      return poolingLayer;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3, 2, 1}};
    }

  }

  public static class Double extends PoolingLayerTest {
    public Double() {
      super(Precision.Double);
    }

  }

  public static class Asymmetric extends PoolingLayerTest {
    public Asymmetric() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setPrecision(precision);
      poolingLayer.setWindowY(4);
      return poolingLayer;
    }

  }

  public static class Float extends PoolingLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }
}
