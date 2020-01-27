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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class BinarySumLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  final int largeSize;
  final int smallSize;

  public BinarySumLayerTest(final Precision precision) {
    this.precision = precision;
    smallSize = 2;
    largeSize = 1200;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, 1}, {smallSize, smallSize, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    BinarySumLayer temp_69_0002 = new BinarySumLayer();
    temp_69_0002.setPrecision(precision);
    BinarySumLayer temp_69_0001 = RefUtil.addRef(temp_69_0002);
    temp_69_0002.freeRef();
    return temp_69_0001;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, 1}, {largeSize, largeSize, 1}};
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BinarySumLayerTest addRef() {
    return (BinarySumLayerTest) super.addRef();
  }

  public static class Double_List extends BinarySumLayerTest {
    public Double_List() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return RefIntStream.range(0, 5).mapToObj(i -> new int[]{smallSize, smallSize, 2}).toArray(i -> new int[i][]);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return RefIntStream.range(0, 5).mapToObj(i -> new int[]{largeSize, largeSize, 3}).toArray(i -> new int[i][]);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

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

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return BinarySumLayer.class;
    }

    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      RefUtil.freeRef(network.add(new BinarySumLayer(), input.addRef(),
          input.addRef()));
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
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    OnePlusOne addRef() {
      return (OnePlusOne) super.addRef();
    }
  }

  public static class Double_Add extends BinarySumLayerTest {
    public Double_Add() {
      super(Precision.Double);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double_Add addRef() {
      return (Double_Add) super.addRef();
    }
  }

  public static class Double_Subtract extends BinarySumLayerTest {
    public Double_Subtract() {
      super(Precision.Double);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Double_Subtract[] addRefs(@Nullable Double_Subtract[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double_Subtract::addRef)
          .toArray((x) -> new Double_Subtract[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      BinarySumLayer temp_69_0004 = new BinarySumLayer(1.0, -1.0);
      temp_69_0004.setPrecision(precision);
      BinarySumLayer temp_69_0003 = RefUtil.addRef(temp_69_0004);
      temp_69_0004.freeRef();
      return temp_69_0003;
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Double_Subtract addRef() {
      return (Double_Subtract) super.addRef();
    }
  }

  public static class Float_Add extends BinarySumLayerTest {
    public Float_Add() {
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
    Float_Add addRef() {
      return (Float_Add) super.addRef();
    }
  }

  public static class Float_Avg extends BinarySumLayerTest {
    public Float_Avg() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      BinarySumLayer temp_69_0006 = new BinarySumLayer(0.5, 0.5);
      temp_69_0006.setPrecision(precision);
      BinarySumLayer temp_69_0005 = RefUtil.addRef(temp_69_0006);
      temp_69_0006.freeRef();
      return temp_69_0005;
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Float_Avg addRef() {
      return (Float_Avg) super.addRef();
    }
  }
}
