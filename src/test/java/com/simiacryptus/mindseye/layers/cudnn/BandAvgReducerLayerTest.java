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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class BandAvgReducerLayerTest extends CudnnLayerTestBase {

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
    BandReducerLayer temp_72_0002 = new BandReducerLayer();
    BandReducerLayer temp_72_0005 = temp_72_0002
        .setMode(PoolingLayer.PoolingMode.Avg);
    BandReducerLayer temp_72_0006 = temp_72_0005.setAlpha(alpha);
    BandReducerLayer temp_72_0001 = temp_72_0006.setPrecision(precision);
    if (null != temp_72_0006)
      temp_72_0006.freeRef();
    if (null != temp_72_0005)
      temp_72_0005.freeRef();
    if (null != temp_72_0002)
      temp_72_0002.freeRef();
    return temp_72_0001;
  }

  public static @SuppressWarnings("unused")
  BandAvgReducerLayerTest[] addRefs(BandAvgReducerLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayerTest::addRef)
        .toArray((x) -> new BandAvgReducerLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  BandAvgReducerLayerTest[][] addRefs(BandAvgReducerLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandAvgReducerLayerTest::addRefs)
        .toArray((x) -> new BandAvgReducerLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    BandAvgReducerLayer temp_72_0004 = new BandAvgReducerLayer();
    BandAvgReducerLayer temp_72_0007 = temp_72_0004.setAlpha(alpha);
    BandAvgReducerLayer temp_72_0003 = temp_72_0007.setPrecision(precision);
    if (null != temp_72_0007)
      temp_72_0007.freeRef();
    if (null != temp_72_0004)
      temp_72_0004.freeRef();
    return temp_72_0003;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, 3}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  BandAvgReducerLayerTest addRef() {
    return (BandAvgReducerLayerTest) super.addRef();
  }

  public static @RefAware
  class Double extends BandAvgReducerLayerTest {
    public Double() {
      super(Precision.Double, 1.0, 8, 1200);
    }

    public static @SuppressWarnings("unused")
    Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Double addRef() {
      return (Double) super.addRef();
    }
  }

  public static @RefAware
  class Negative extends BandAvgReducerLayerTest {
    public Negative() {
      super(Precision.Double, -5.0, 8, 1200);
    }

    public static @SuppressWarnings("unused")
    Negative[] addRefs(Negative[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Negative::addRef)
          .toArray((x) -> new Negative[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Negative addRef() {
      return (Negative) super.addRef();
    }
  }

  public static @RefAware
  class Asymmetric extends BandAvgReducerLayerTest {
    public Asymmetric() {
      super(Precision.Double, 1.0, 4, 1200);
    }

    public static @SuppressWarnings("unused")
    Asymmetric[] addRefs(Asymmetric[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Asymmetric::addRef)
          .toArray((x) -> new Asymmetric[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{3, 5, 2}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{1200, 800, 3}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Asymmetric addRef() {
      return (Asymmetric) super.addRef();
    }

  }

  public static @RefAware
  class Float extends BandAvgReducerLayerTest {
    public Float() {
      super(Precision.Float, 1.0, 4, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public static @SuppressWarnings("unused")
    Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Float addRef() {
      return (Float) super.addRef();
    }
  }
}
