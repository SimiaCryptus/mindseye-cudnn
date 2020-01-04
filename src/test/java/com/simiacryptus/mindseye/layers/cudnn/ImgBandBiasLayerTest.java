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
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class ImgBandBiasLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public ImgBandBiasLayerTest(final Precision precision) {
    this.precision = precision;
  }

  public static @SuppressWarnings("unused")
  ImgBandBiasLayerTest[] addRefs(ImgBandBiasLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayerTest::addRef)
        .toArray((x) -> new ImgBandBiasLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgBandBiasLayerTest[][] addRefs(ImgBandBiasLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayerTest::addRefs)
        .toArray((x) -> new ImgBandBiasLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgBandBiasLayer(3).setPrecision(precision).addWeights(this::random);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{1200, 1200, 3}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgBandBiasLayerTest addRef() {
    return (ImgBandBiasLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Double extends ImgBandBiasLayerTest {
    public Double() {
      super(Precision.Double);
    }

    public static @SuppressWarnings("unused")
    Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
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

  public static @com.simiacryptus.ref.lang.RefAware
  class Float extends ImgBandBiasLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public static @SuppressWarnings("unused")
    Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
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
