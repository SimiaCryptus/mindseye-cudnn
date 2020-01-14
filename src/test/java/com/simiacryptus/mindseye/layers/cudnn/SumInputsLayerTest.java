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
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class SumInputsLayerTest extends CudnnLayerTestBase {

  private static int largeSize;
  final Precision precision;
  final int inputBands;
  final int inputs;

  public SumInputsLayerTest(final Precision precision, int inputBands, int inputs, final int largeSize) {
    this.precision = precision;
    this.inputBands = inputBands;
    this.inputs = inputs;
    SumInputsLayerTest.largeSize = largeSize;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.SumInputsLayer.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SumInputsLayerTest[] addRefs(@Nullable SumInputsLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayerTest::addRef)
        .toArray((x) -> new SumInputsLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SumInputsLayerTest[][] addRefs(@Nullable SumInputsLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayerTest::addRefs)
        .toArray((x) -> new SumInputsLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return RefIntStream.range(0, inputs).mapToObj(i -> new int[]{2, 2, inputBands}).toArray(i -> new int[i][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer temp_73_0002 = new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer();
    com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer temp_73_0001 = temp_73_0002.setPrecision(precision);
    temp_73_0002.freeRef();
    return temp_73_0001;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return RefIntStream.range(0, inputs).mapToObj(i -> new int[]{largeSize, largeSize, inputBands})
        .toArray(i -> new int[i][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumInputsLayerTest addRef() {
    return (SumInputsLayerTest) super.addRef();
  }

  public static class Double_List extends SumInputsLayerTest {
    public Double_List() {
      super(Precision.Double, 1, 5, 1200);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Double_List[] addRefs(@Nullable Double_List[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double_List::addRef).toArray((x) -> new Double_List[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double_List addRef() {
      return (Double_List) super.addRef();
    }

  }

  public static class OnePlusOne extends CudnnLayerTestBase {

    public OnePlusOne() {
      super();
    }

    @Override
    public Layer getReferenceLayer() {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      RefUtil.freeRef(network.add(new SumInputsLayer(), input.addRef(),
          input.addRef()));
      input.freeRef();
      return network;
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.SumInputsLayer.class;
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    OnePlusOne[] addRefs(@Nullable OnePlusOne[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(OnePlusOne::addRef).toArray((x) -> new OnePlusOne[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      RefUtil.freeRef(network.add(new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer(),
          input.addRef(), input.addRef()));
      input.freeRef();
      return network;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{1200, 800, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    OnePlusOne addRef() {
      return (OnePlusOne) super.addRef();
    }

  }

  public static class Big_Double_Add extends Big {
    public Big_Double_Add() {
      super(Precision.Double, 256, 8, 100);
      testingBatchSize = 2;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Big_Double_Add[] addRefs(@Nullable Big_Double_Add[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big_Double_Add::addRef)
          .toArray((x) -> new Big_Double_Add[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Big_Double_Add addRef() {
      return (Big_Double_Add) super.addRef();
    }
  }

  public static class Double_Add extends SumInputsLayerTest {
    public Double_Add() {
      super(Precision.Double, 1, 2, 1200);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Double_Add[] addRefs(@Nullable Double_Add[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double_Add::addRef).toArray((x) -> new Double_Add[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double_Add addRef() {
      return (Double_Add) super.addRef();
    }
  }

  public static class Float_Add extends SumInputsLayerTest {
    public Float_Add() {
      super(Precision.Float, 1, 2, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Float_Add[] addRefs(@Nullable Float_Add[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Float_Add::addRef).toArray((x) -> new Float_Add[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Float_Add addRef() {
      return (Float_Add) super.addRef();
    }

  }

  public abstract static class Big extends SumInputsLayerTest {

    public Big(final Precision precision, int inputBands, int inputs, final int largeSize) {
      super(precision, inputBands, inputs, largeSize);
      validateDifferentials = false;
      setTestTraining(false);
      testingBatchSize = 5;
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

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Big[] addRefs(@Nullable Big[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big::addRef).toArray((x) -> new Big[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Big addRef() {
      return (Big) super.addRef();
    }

  }

}
