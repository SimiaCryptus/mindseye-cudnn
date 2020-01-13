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

public abstract class PoolingLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public PoolingLayerTest(final Precision precision) {
    this.precision = precision;
  }

  public static @SuppressWarnings("unused") PoolingLayerTest[] addRefs(PoolingLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PoolingLayerTest::addRef)
        .toArray((x) -> new PoolingLayerTest[x]);
  }

  public static @SuppressWarnings("unused") PoolingLayerTest[][] addRefs(PoolingLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PoolingLayerTest::addRefs)
        .toArray((x) -> new PoolingLayerTest[x][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    PoolingLayer temp_60_0002 = new PoolingLayer();
    PoolingLayer temp_60_0001 = temp_60_0002.setPrecision(precision);
    if (null != temp_60_0002)
      temp_60_0002.freeRef();
    return temp_60_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 8, 8, 1 } };
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 800, 800, 16 } };
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") PoolingLayerTest addRef() {
    return (PoolingLayerTest) super.addRef();
  }

  public static class Repro extends PoolingLayerTest {
    public Repro() {
      super(Precision.Double);
    }

    public static @SuppressWarnings("unused") Repro[] addRefs(Repro[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Repro::addRef).toArray((x) -> new Repro[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3, 2, 1 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 3, 2, 512 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      PoolingLayer temp_60_0004 = new PoolingLayer();
      PoolingLayer temp_60_0007 = temp_60_0004.setWindowXY(3, 2);
      PoolingLayer temp_60_0008 = temp_60_0007.setStrideXY(3, 2);
      PoolingLayer temp_60_0003 = temp_60_0008.setPrecision(precision);
      if (null != temp_60_0008)
        temp_60_0008.freeRef();
      if (null != temp_60_0007)
        temp_60_0007.freeRef();
      if (null != temp_60_0004)
        temp_60_0004.freeRef();
      return temp_60_0003;
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Repro addRef() {
      return (Repro) super.addRef();
    }

  }

  public static class Double extends PoolingLayerTest {
    public Double() {
      super(Precision.Double);
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

  public static class Asymmetric extends PoolingLayerTest {
    public Asymmetric() {
      super(Precision.Double);
    }

    public static @SuppressWarnings("unused") Asymmetric[] addRefs(Asymmetric[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Asymmetric::addRef).toArray((x) -> new Asymmetric[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      PoolingLayer temp_60_0006 = new PoolingLayer();
      PoolingLayer temp_60_0009 = temp_60_0006.setPrecision(precision);
      PoolingLayer temp_60_0005 = temp_60_0009.setWindowY(4);
      if (null != temp_60_0009)
        temp_60_0009.freeRef();
      if (null != temp_60_0006)
        temp_60_0006.freeRef();
      return temp_60_0005;
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Asymmetric addRef() {
      return (Asymmetric) super.addRef();
    }

  }

  public static class Float extends PoolingLayerTest {
    public Float() {
      super(Precision.Float);
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
