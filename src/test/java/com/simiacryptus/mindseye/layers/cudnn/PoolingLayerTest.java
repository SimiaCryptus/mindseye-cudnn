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

public abstract class PoolingLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public PoolingLayerTest(final Precision precision) {
    this.precision = precision;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  PoolingLayerTest[] addRefs(@Nullable PoolingLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PoolingLayerTest::addRef)
        .toArray((x) -> new PoolingLayerTest[x]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    PoolingLayer temp_60_0002 = new PoolingLayer();
    temp_60_0002.setPrecision(precision);
    PoolingLayer temp_60_0001 = RefUtil.addRef(temp_60_0002);
    temp_60_0002.freeRef();
    return temp_60_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{800, 800, 16}};
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PoolingLayerTest addRef() {
    return (PoolingLayerTest) super.addRef();
  }

  public static class Repro extends PoolingLayerTest {
    public Repro() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{3, 2, 1}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{3, 2, 512}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      PoolingLayer temp_60_0004 = new PoolingLayer();
      temp_60_0004.setWindowXY(3, 2);
      PoolingLayer temp_60_0007 = temp_60_0004.addRef();
      temp_60_0007.setStrideXY(3, 2);
      PoolingLayer temp_60_0008 = temp_60_0007.addRef();
      temp_60_0008.setPrecision(precision);
      PoolingLayer temp_60_0003 = RefUtil.addRef(temp_60_0008);
      temp_60_0008.freeRef();
      temp_60_0007.freeRef();
      temp_60_0004.freeRef();
      return temp_60_0003;
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Repro addRef() {
      return (Repro) super.addRef();
    }
  }

  public static class Double extends PoolingLayerTest {
    public Double() {
      super(Precision.Double);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double addRef() {
      return (Double) super.addRef();
    }
  }

  public static class Asymmetric extends PoolingLayerTest {
    public Asymmetric() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      PoolingLayer temp_60_0006 = new PoolingLayer();
      temp_60_0006.setPrecision(precision);
      PoolingLayer temp_60_0009 = RefUtil.addRef(temp_60_0006);
      temp_60_0009.setWindowY(4);
      PoolingLayer temp_60_0005 = temp_60_0009.addRef();
      temp_60_0009.freeRef();
      temp_60_0006.freeRef();
      return temp_60_0005;
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Asymmetric addRef() {
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

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Float addRef() {
      return (Float) super.addRef();
    }
  }
}
