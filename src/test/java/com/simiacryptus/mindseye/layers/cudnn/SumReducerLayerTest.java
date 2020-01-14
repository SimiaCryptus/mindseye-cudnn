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
import java.util.Arrays;
import java.util.Random;

public abstract class SumReducerLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public SumReducerLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SumReducerLayerTest[] addRefs(@Nullable SumReducerLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayerTest::addRef)
        .toArray((x) -> new SumReducerLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SumReducerLayerTest[][] addRefs(@Nullable SumReducerLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayerTest::addRefs)
        .toArray((x) -> new SumReducerLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    SumReducerLayer temp_59_0002 = new SumReducerLayer();
    SumReducerLayer temp_59_0001 = temp_59_0002.setPrecision(precision);
    temp_59_0002.freeRef();
    return temp_59_0001;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumReducerLayerTest addRef() {
    return (SumReducerLayerTest) super.addRef();
  }

  public static class Double extends SumReducerLayerTest {
    public Double() {
      super(Precision.Double);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Double[] addRefs(@Nullable Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
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

  public static class Asymmetric extends SumReducerLayerTest {
    public Asymmetric() {
      super(Precision.Double);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Asymmetric[] addRefs(@Nullable Asymmetric[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Asymmetric::addRef).toArray((x) -> new Asymmetric[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{3, 5, 2}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{1000, 600, 3}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Asymmetric addRef() {
      return (Asymmetric) super.addRef();
    }

  }

  public static class Float extends SumReducerLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Float[] addRefs(@Nullable Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
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
