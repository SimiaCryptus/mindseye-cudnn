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
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class NProductLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public NProductLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}, {8, 8, 1}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{1200, 1200, 3}, {1200, 1200, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    NProductLayer temp_54_0002 = new NProductLayer();
    temp_54_0002.setPrecision(precision);
    NProductLayer temp_54_0001 = RefUtil.addRef(temp_54_0002);
    temp_54_0002.freeRef();
    return temp_54_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  NProductLayerTest addRef() {
    return (NProductLayerTest) super.addRef();
  }

  public static class Double extends NProductLayerTest {
    public Double() {
      super(Precision.Double);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double addRef() {
      return (Double) super.addRef();
    }
  }

  public static class Double3 extends NProductLayerTest {
    public Double3() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{8, 8, 1}, {8, 8, 1}, {8, 8, 1}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{1200, 1200, 3}, {1200, 1200, 3}, {1200, 1200, 3}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double3 addRef() {
      return (Double3) super.addRef();
    }
  }

  public static class Float extends NProductLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Float addRef() {
      return (Float) super.addRef();
    }
  }
}
