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

public abstract class ImgBandBiasLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public ImgBandBiasLayerTest(final Precision precision) {
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
    ImgBandBiasLayer temp_55_0002 = new ImgBandBiasLayer(3);
    temp_55_0002.setPrecision(precision);
    ImgBandBiasLayer temp_55_0003 = RefUtil.addRef(temp_55_0002);
    temp_55_0003.addWeights(() -> random());
    ImgBandBiasLayer temp_55_0001 = temp_55_0003.addRef();
    temp_55_0003.freeRef();
    temp_55_0002.freeRef();
    return temp_55_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 3}};
  }

  public static class Double extends ImgBandBiasLayerTest {
    public Double() {
      super(Precision.Double);
    }

  }

  public static class Float extends ImgBandBiasLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
