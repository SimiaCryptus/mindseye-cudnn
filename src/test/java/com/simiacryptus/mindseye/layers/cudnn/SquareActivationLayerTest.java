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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class SquareActivationLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  private final double alpha;

  public SquareActivationLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    PipelineNetwork network = new PipelineNetwork();
    LinearActivationLayer temp_65_0001 = new LinearActivationLayer();
    NthPowerActivationLayer temp_65_0002 = new NthPowerActivationLayer();
    temp_65_0001.setScale(alpha);
    temp_65_0002.setPower(2);
    RefUtil
        .freeRef(network.add(temp_65_0001.addRef(), network.add(temp_65_0002.addRef(), network.getInput(0))));
    temp_65_0002.freeRef();
    temp_65_0001.freeRef();
    return network;
    //return new NthPowerActivationLayer().setPower(2);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{4, 4, 1}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    SquareActivationLayer temp_65_0004 = new SquareActivationLayer();
    temp_65_0004.setPrecision(precision);
    SquareActivationLayer temp_65_0005 = RefUtil.addRef(temp_65_0004);
    temp_65_0005.setAlpha(alpha);
    SquareActivationLayer temp_65_0003 = temp_65_0005.addRef();
    temp_65_0005.freeRef();
    temp_65_0004.freeRef();
    return temp_65_0003;
  }

  public static class Double extends SquareActivationLayerTest {
    public Double() {
      super(Precision.Double, 1.0);
    }

  }

  public static class Negative extends SquareActivationLayerTest {
    public Negative() {
      super(Precision.Double, -1.0);
    }

  }

  public static class Float extends SquareActivationLayerTest {
    public Float() {
      super(Precision.Float, 1.0);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }
}
