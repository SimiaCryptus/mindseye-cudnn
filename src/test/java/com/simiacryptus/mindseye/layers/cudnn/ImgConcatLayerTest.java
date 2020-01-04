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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;

public abstract @com.simiacryptus.ref.lang.RefAware class ImgConcatLayerTest extends CudnnLayerTestBase {

  private final Precision precision;
  private final int[] bandSeq;
  private final int smallSize;
  private final int largeSize;

  public ImgConcatLayerTest(final Precision precision, int inputs, int bandsPerInput, final int smallSize,
      final int largeSize) {
    this(precision, com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputs).map(i -> bandsPerInput).toArray(),
        smallSize, largeSize);
  }

  public ImgConcatLayerTest(final Precision precision, final int[] bandSeq, final int smallSize, final int largeSize) {
    this.bandSeq = bandSeq;
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return com.simiacryptus.ref.wrappers.RefArrays.stream(bandSeq).mapToObj(x -> new int[] { smallSize, smallSize, x })
        .toArray(i -> new int[i][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgConcatLayer().setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return com.simiacryptus.ref.wrappers.RefArrays.stream(bandSeq).mapToObj(x -> new int[] { largeSize, largeSize, x })
        .toArray(i -> new int[i][]);
  }

  public static @com.simiacryptus.ref.lang.RefAware class BandLimitTest extends ImgConcatLayerTest {

    public BandLimitTest() {
      super(Precision.Double, 2, 1, 8, 1200);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 1, 1, 3 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(2);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") BandLimitTest addRef() {
      return (BandLimitTest) super.addRef();
    }

    public static @SuppressWarnings("unused") BandLimitTest[] addRefs(BandLimitTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandLimitTest::addRef)
          .toArray((x) -> new BandLimitTest[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class BandConcatLimitTest extends ImgConcatLayerTest {

    public BandConcatLimitTest() {
      super(Precision.Double, new int[] { 2, 3, 4 }, 2, 1200);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(8);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") BandConcatLimitTest addRef() {
      return (BandConcatLimitTest) super.addRef();
    }

    public static @SuppressWarnings("unused") BandConcatLimitTest[] addRefs(BandConcatLimitTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandConcatLimitTest::addRef)
          .toArray((x) -> new BandConcatLimitTest[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class Double extends ImgConcatLayerTest {
    public Double() {
      super(Precision.Double, 2, 1, 8, 1200);
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
  //
  //  /**
  //   * Basic 64-bit apply
  //   */
  //  public static class BigDouble extends Big {
  //    /**
  //     * Instantiates a new Double.
  //     */
  //    public BigDouble() {
  //      super(Precision.Double, 8, 512);
  //    }
  //  }

  public abstract static @com.simiacryptus.ref.lang.RefAware class Big extends ImgConcatLayerTest {

    public Big(final Precision precision, final int inputs, final int bandsPerInput) {
      super(precision, inputs, bandsPerInput, 8, 1200);
      this.validateDifferentials = false;
      setTestTraining(false);
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution)
        return null;
      return (new BatchingTester(1e-2, true) {
        @Override
        public double getRandom() {
          return random();
        }

        public @SuppressWarnings("unused") void _free() {
        }
      }).setBatchSize(5);
    }

    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Big addRef() {
      return (Big) super.addRef();
    }

    public static @SuppressWarnings("unused") Big[] addRefs(Big[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Big::addRef).toArray((x) -> new Big[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Float extends ImgConcatLayerTest {
    public Float() {
      super(Precision.Float, 2, 1, 8, 1200);
      tolerance = 1e-2;
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

  public @Override @SuppressWarnings("unused") ImgConcatLayerTest addRef() {
    return (ImgConcatLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgConcatLayerTest[] addRefs(ImgConcatLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayerTest::addRef)
        .toArray((x) -> new ImgConcatLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ImgConcatLayerTest[][] addRefs(ImgConcatLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayerTest::addRefs)
        .toArray((x) -> new ImgConcatLayerTest[x][]);
  }

}
