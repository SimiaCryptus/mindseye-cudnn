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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class SquareActivationLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  private final double alpha;

  public SquareActivationLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    PipelineNetwork network = new PipelineNetwork();
    network.add(new LinearActivationLayer().setScale(alpha),
        network.add(new NthPowerActivationLayer().setPower(2), network.getInput(0)));
    return network;
    //return new NthPowerActivationLayer().setPower(2);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 4, 4, 1 } };
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][] { { 1200, 1200, 3 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new SquareActivationLayer().setPrecision(precision).setAlpha(alpha);
  }

  public static @com.simiacryptus.ref.lang.RefAware class Double extends SquareActivationLayerTest {
    public Double() {
      super(Precision.Double, 1.0);
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

  public static @com.simiacryptus.ref.lang.RefAware class Negative extends SquareActivationLayerTest {
    public Negative() {
      super(Precision.Double, -1.0);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Negative addRef() {
      return (Negative) super.addRef();
    }

    public static @SuppressWarnings("unused") Negative[] addRefs(Negative[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Negative::addRef)
          .toArray((x) -> new Negative[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class Float extends SquareActivationLayerTest {
    public Float() {
      super(Precision.Float, 1.0);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
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

  public @Override @SuppressWarnings("unused") SquareActivationLayerTest addRef() {
    return (SquareActivationLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") SquareActivationLayerTest[] addRefs(SquareActivationLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SquareActivationLayerTest::addRef)
        .toArray((x) -> new SquareActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused") SquareActivationLayerTest[][] addRefs(SquareActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SquareActivationLayerTest::addRefs)
        .toArray((x) -> new SquareActivationLayerTest[x][]);
  }
}
