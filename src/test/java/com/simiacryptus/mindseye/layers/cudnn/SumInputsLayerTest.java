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
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Sum inputs layer test.
 */
public abstract class SumInputsLayerTest extends CudnnLayerTestBase {

  private static int largeSize;
  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Input bands.
   */
  final int inputBands;
  /**
   * The Inputs.
   */
  final int inputs;

  /**
   * Instantiates a new Sum inputs layer test.
   *
   * @param precision  the precision
   * @param inputBands the input bands
   * @param inputs     the inputs
   * @param largeSize  the large size
   */
  public SumInputsLayerTest(final Precision precision, int inputBands, int inputs, final int largeSize) {
    this.precision = precision;
    this.inputBands = inputBands;
    this.inputs = inputs;
    SumInputsLayerTest.largeSize = largeSize;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return RefIntStream.range(0, inputs).mapToObj(i -> new int[]{largeSize, largeSize, inputBands})
        .toArray(i -> new int[i][]);
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer temp_73_0002 = new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer();
    temp_73_0002.setPrecision(precision);
    com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer temp_73_0001 = RefUtil.addRef(temp_73_0002);
    temp_73_0002.freeRef();
    return temp_73_0001;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.SumInputsLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return RefIntStream.range(0, inputs).mapToObj(i -> new int[]{2, 2, inputBands}).toArray(i -> new int[i][]);
  }

  /**
   * The type Double list.
   */
  public static class Double_List extends SumInputsLayerTest {
    /**
     * Instantiates a new Double list.
     */
    public Double_List() {
      super(Precision.Double, 1, 5, 1200);
    }

  }

  /**
   * The type One plus one.
   */
  public static class OnePlusOne extends CudnnLayerTestBase {

    /**
     * Instantiates a new One plus one.
     */
    public OnePlusOne() {
      super();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{1200, 800, 1}};
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      RefUtil.freeRef(network.add(new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer(),
          input.addRef(), input.addRef()));
      input.freeRef();
      return network;
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
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }

  }

  /**
   * The type Big double add.
   */
  public static class Big_Double_Add extends Big {
    /**
     * Instantiates a new Big double add.
     */
    public Big_Double_Add() {
      super(Precision.Double, 256, 8, 100);
      testingBatchSize = 2;
    }

  }

  /**
   * The type Double add.
   */
  public static class Double_Add extends SumInputsLayerTest {
    /**
     * Instantiates a new Double add.
     */
    public Double_Add() {
      super(Precision.Double, 1, 2, 1200);
    }

  }

  /**
   * The type Float add.
   */
  public static class Float_Add extends SumInputsLayerTest {
    /**
     * Instantiates a new Float add.
     */
    public Float_Add() {
      super(Precision.Float, 1, 2, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }

  /**
   * The type Big.
   */
  public abstract static class Big extends SumInputsLayerTest {

    /**
     * Instantiates a new Big.
     *
     * @param precision  the precision
     * @param inputBands the input bands
     * @param inputs     the inputs
     * @param largeSize  the large size
     */
    public Big(final Precision precision, int inputBands, int inputs, final int largeSize) {
      super(precision, inputBands, inputs, largeSize);
      testingBatchSize = 5;
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, false, this.testingBatchSize);
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
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

}
