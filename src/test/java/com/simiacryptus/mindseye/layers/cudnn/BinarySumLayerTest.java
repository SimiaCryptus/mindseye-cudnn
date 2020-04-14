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

/**
 * The type Binary sum layer test.
 */
public abstract class BinarySumLayerTest extends CudnnLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Large size.
   */
  final int largeSize;
  /**
   * The Small size.
   */
  final int smallSize;

  /**
   * Instantiates a new Binary sum layer test.
   *
   * @param precision the precision
   */
  public BinarySumLayerTest(final Precision precision) {
    this.precision = precision;
    smallSize = 2;
    largeSize = 600;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, 1}, {largeSize, largeSize, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    BinarySumLayer binarySumLayer = new BinarySumLayer();
    binarySumLayer.setPrecision(precision);
    return binarySumLayer;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, 1}, {smallSize, smallSize, 1}};
  }

  /**
   * The type Double list.
   */
  public static class Double_List extends BinarySumLayerTest {
    /**
     * Instantiates a new Double list.
     */
    public Double_List() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return RefIntStream.range(0, 5).mapToObj(i -> new int[]{largeSize, largeSize, 3}).toArray(i -> new int[i][]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return RefIntStream.range(0, 5).mapToObj(i -> new int[]{smallSize, smallSize, 2}).toArray(i -> new int[i][]);
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
      RefUtil.freeRef(network.add(new BinarySumLayer(), input.addRef(),
          input.addRef()));
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

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{4, 4, 1}};
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return BinarySumLayer.class;
    }
  }

  /**
   * The type Double add.
   */
  public static class Double_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Double add.
     */
    public Double_Add() {
      super(Precision.Double);
    }
  }

  /**
   * The type Double subtract.
   */
  public static class Double_Subtract extends BinarySumLayerTest {
    /**
     * Instantiates a new Double subtract.
     */
    public Double_Subtract() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      BinarySumLayer binarySumLayer = new BinarySumLayer(1.0, -1.0);
      binarySumLayer.setPrecision(precision);
      return binarySumLayer;
    }

  }

  /**
   * The type Float add.
   */
  public static class Float_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Float add.
     */
    public Float_Add() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }

  /**
   * The type Float avg.
   */
  public static class Float_Avg extends BinarySumLayerTest {
    /**
     * Instantiates a new Float avg.
     */
    public Float_Avg() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      BinarySumLayer binarySumLayer = new BinarySumLayer(0.5, 0.5);
      binarySumLayer.setPrecision(precision);
      return binarySumLayer;
    }
  }
}
