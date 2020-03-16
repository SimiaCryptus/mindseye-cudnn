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
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public abstract class ImgConcatLayerTest extends CudnnLayerTestBase {

  private final Precision precision;
  private final int[] bandSeq;
  private final int smallSize;
  private final int largeSize;

  public ImgConcatLayerTest(final Precision precision, int inputs, int bandsPerInput, final int smallSize,
                            final int largeSize) {
    this(precision, RefIntStream.range(0, inputs).map(i -> bandsPerInput).toArray(), smallSize, largeSize);
  }

  public ImgConcatLayerTest(final Precision precision, final int[] bandSeq, final int smallSize, final int largeSize) {
    this.bandSeq = bandSeq;
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return RefArrays.stream(bandSeq).mapToObj(x -> new int[]{largeSize, largeSize, x}).toArray(i -> new int[i][]);
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    ImgConcatLayer imgConcatLayer = new ImgConcatLayer();
    imgConcatLayer.setPrecision(precision);
    return imgConcatLayer;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class;
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

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return RefArrays.stream(bandSeq).mapToObj(x -> new int[]{smallSize, smallSize, x}).toArray(i -> new int[i][]);
  }

  public static class BandLimitTest extends ImgConcatLayerTest {

    public BandLimitTest() {
      super(Precision.Double, 2, 1, 8, 1200);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return getSmallDims();
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgConcatLayer imgConcatLayer = new ImgConcatLayer();
      imgConcatLayer.setMaxBands(2);
      return imgConcatLayer;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{1, 1, 3}};
    }

  }

  public static class BandConcatLimitTest extends ImgConcatLayerTest {

    public BandConcatLimitTest() {
      super(Precision.Double, new int[]{2, 3, 4}, 2, 1200);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return getSmallDims();
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      ImgConcatLayer imgConcatLayer = new ImgConcatLayer();
      imgConcatLayer.setMaxBands(8);
      return imgConcatLayer;
    }

  }

  public static class Double extends ImgConcatLayerTest {
    public Double() {
      super(Precision.Double, 2, 1, 8, 1200);
    }

  }

  public abstract static class Big extends ImgConcatLayerTest {

    public Big(final Precision precision, final int inputs, final int bandsPerInput) {
      super(precision, inputs, bandsPerInput, 8, 1200);
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, true, 5);
    }

    @Override
    @Disabled
    public void derivativeTest() {
      super.derivativeTest();
    }

    @Override
    @Disabled
    public void trainingTest() {
      super.trainingTest();
    }

    @Override
    @Disabled
    public void jsonTest() {
      super.jsonTest();
    }

    @Override
    @Disabled
    public void perfTest() {
      super.perfTest();
    }

  }

  public static class Float extends ImgConcatLayerTest {
    public Float() {
      super(Precision.Float, 2, 1, 8, 1200);
      tolerance = 1e-2;
    }

  }

}
