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
import java.util.Arrays;
import java.util.Random;

public abstract class BandReducerLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  private final double alpha;

  public BandReducerLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
  }

  public static @SuppressWarnings("unused") BandReducerLayerTest[] addRefs(BandReducerLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandReducerLayerTest::addRef)
        .toArray((x) -> new BandReducerLayerTest[x]);
  }

  public static @SuppressWarnings("unused") BandReducerLayerTest[][] addRefs(BandReducerLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BandReducerLayerTest::addRefs)
        .toArray((x) -> new BandReducerLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 2, 2, 1 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    BandReducerLayer temp_66_0002 = new BandReducerLayer();
    BandReducerLayer temp_66_0003 = temp_66_0002.setAlpha(alpha);
    BandReducerLayer temp_66_0001 = temp_66_0003.setPrecision(precision);
    if (null != temp_66_0003)
      temp_66_0003.freeRef();
    if (null != temp_66_0002)
      temp_66_0002.freeRef();
    return temp_66_0001;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 32, 32, 3 } };
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BandReducerLayerTest addRef() {
    return (BandReducerLayerTest) super.addRef();
  }

  public static class Double extends BandReducerLayerTest {
    public Double() {
      super(Precision.Double, 1.0);
    }

    public static @SuppressWarnings("unused") Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Double addRef() {
      return (Double) super.addRef();
    }
  }

  public static class Negative extends BandReducerLayerTest {
    public Negative() {
      super(Precision.Double, -5.0);
    }

    public static @SuppressWarnings("unused") Negative[] addRefs(Negative[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Negative::addRef).toArray((x) -> new Negative[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Negative addRef() {
      return (Negative) super.addRef();
    }
  }

  public static class Asymmetric extends BandReducerLayerTest {
    public Asymmetric() {
      super(Precision.Double, 1.0);
    }

    public static @SuppressWarnings("unused") Asymmetric[] addRefs(Asymmetric[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Asymmetric::addRef).toArray((x) -> new Asymmetric[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3, 5, 2 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 200, 100, 3 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Asymmetric addRef() {
      return (Asymmetric) super.addRef();
    }

  }

  public static class Float extends BandReducerLayerTest {
    public Float() {
      super(Precision.Float, 1.0);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public static @SuppressWarnings("unused") Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Float addRef() {
      return (Float) super.addRef();
    }
  }
}
