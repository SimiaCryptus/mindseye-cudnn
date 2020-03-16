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

public abstract class GateBiasLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public GateBiasLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 3}, {1, 1, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    GateBiasLayer temp_74_0002 = new GateBiasLayer();
    temp_74_0002.setPrecision(precision);
    GateBiasLayer temp_74_0001 = RefUtil.addRef(temp_74_0002);
    temp_74_0002.freeRef();
    return temp_74_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{4, 4, 3}, {1, 1, 3}};
  }

  public static class Double extends GateBiasLayerTest {
    public Double() {
      super(Precision.Double);
    }
  }

  public static class Float extends GateBiasLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
