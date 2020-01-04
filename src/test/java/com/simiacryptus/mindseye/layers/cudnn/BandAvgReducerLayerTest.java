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
import javax.annotation.Nullable;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class BandAvgReducerLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  private final double alpha;
  private final int smallSize;
  private final int largeSize;

  public BandAvgReducerLayerTest(final Precision precision, final double alpha, final int smallSize,
      final int largeSize) {
    this.precision = precision;
    this.alpha = alpha;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).setAlpha(alpha).setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { smallSize, smallSize, 1 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BandAvgReducerLayer().setAlpha(alpha).setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { largeSize, largeSize, 3 } };
  }

  public static @com.simiacryptus.ref.lang.RefAware class Double extends BandAvgReducerLayerTest {
    public Double() {
      super(Precision.Double, 1.0, 8, 1200);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Double addRef() {
      return (Double) super.addRef();
    }

    public static @SuppressWarnings("unused") Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class Negative extends BandAvgReducerLayerTest {
    public Negative() {
      super(Precision.Double, -5.0, 8, 1200);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Negative addRef() {
      return (Negative) super.addRef();
    }

    public static @SuppressWarnings("unused") Negative[] addRefs(Negative[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Negative::addRef)
          .toArray((x) -> new Negative[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class Asymmetric extends BandAvgReducerLayerTest {
    public Asymmetric() {
      super(Precision.Double, 1.0, 4, 1200);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3, 5, 2 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 1200, 800, 3 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Asymmetric addRef() {
      return (Asymmetric) super.addRef();
    }

    public static @SuppressWarnings("unused") Asymmetric[] addRefs(Asymmetric[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Asymmetric::addRef)
          .toArray((x) -> new Asymmetric[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Float extends BandAvgReducerLayerTest {
    public Float() {
      super(Precision.Float, 1.0, 4, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Float addRef() {
      return (Float) super.addRef();
    }

    public static @SuppressWarnings("unused") Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BandAvgReducerLayerTest addRef() {
    return (BandAvgReducerLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") BandAvgReducerLayerTest[] addRefs(BandAvgReducerLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayerTest::addRef)
        .toArray((x) -> new BandAvgReducerLayerTest[x]);
  }

  public static @SuppressWarnings("unused") BandAvgReducerLayerTest[][] addRefs(BandAvgReducerLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayerTest::addRefs)
        .toArray((x) -> new BandAvgReducerLayerTest[x][]);
  }
}
