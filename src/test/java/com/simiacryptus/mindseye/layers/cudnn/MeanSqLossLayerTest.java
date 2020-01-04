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

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class MeanSqLossLayerTest extends CudnnLayerTestBase {

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.MeanSqLossLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 8, 8, 1 }, { 8, 8, 1 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new MeanSqLossLayer();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 1200, 1200, 3 }, { 1200, 1200, 3 } };
  }

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends MeanSqLossLayerTest {

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Asymetric extends MeanSqLossLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 2, 3, 1 }, { 2, 3, 1 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 200, 300, 100 }, { 200, 300, 100 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Asymetric addRef() {
      return (Asymetric) super.addRef();
    }

    public static @SuppressWarnings("unused") Asymetric[] addRefs(Asymetric[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Asymetric::addRef)
          .toArray((x) -> new Asymetric[x]);
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") MeanSqLossLayerTest addRef() {
    return (MeanSqLossLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") MeanSqLossLayerTest[] addRefs(MeanSqLossLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayerTest::addRef)
        .toArray((x) -> new MeanSqLossLayerTest[x]);
  }

  public static @SuppressWarnings("unused") MeanSqLossLayerTest[][] addRefs(MeanSqLossLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayerTest::addRefs)
        .toArray((x) -> new MeanSqLossLayerTest[x][]);
  }

}
