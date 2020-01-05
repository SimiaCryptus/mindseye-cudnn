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

public abstract @RefAware
class ProductLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public ProductLayerTest(final Precision precision) {
    this.precision = precision;
  }

  public static @SuppressWarnings("unused")
  ProductLayerTest[] addRefs(ProductLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayerTest::addRef)
        .toArray((x) -> new ProductLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ProductLayerTest[][] addRefs(ProductLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayerTest::addRefs)
        .toArray((x) -> new ProductLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{4, 4, 3}, {1, 1, 3}};
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{400, 400, 30}, {1, 1, 30}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    ProductLayer temp_56_0002 = new ProductLayer();
    ProductLayer temp_56_0001 = temp_56_0002.setPrecision(precision);
    if (null != temp_56_0002)
      temp_56_0002.freeRef();
    return temp_56_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ProductLayerTest addRef() {
    return (ProductLayerTest) super.addRef();
  }

  public static @RefAware
  class Mask extends ProductLayerTest {
    public Mask() {
      super(Precision.Double);
    }

    public static @SuppressWarnings("unused")
    Mask[] addRefs(Mask[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Mask::addRef).toArray((x) -> new Mask[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 3}, {4, 4, 1}};
    }

    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{400, 400, 30}, {400, 400, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Mask addRef() {
      return (Mask) super.addRef();
    }
  }

  public static @RefAware
  class Double extends ProductLayerTest {
    public Double() {
      super(Precision.Double);
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
  class Float extends ProductLayerTest {
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
