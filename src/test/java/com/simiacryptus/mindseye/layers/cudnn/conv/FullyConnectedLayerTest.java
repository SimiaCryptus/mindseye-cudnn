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

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.CudnnLayerTestBase;
import com.simiacryptus.mindseye.layers.java.FullyConnectedReferenceLayer;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Fully connected layer test.
 */
public abstract class FullyConnectedLayerTest extends CudnnLayerTestBase {

  /**
   * The Input dim.
   */
  @Nonnull
  protected final int[] inputDim;
  /**
   * The Fully connected layer.
   */
  @Nonnull
  @RefIgnore
  protected final FullyConnectedLayer fullyConnectedLayer;
  /**
   * The Layer.
   */
  @Nonnull
  @RefIgnore
  protected final Layer layer;

  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   * @param batchBands the batch bands
   */
  public FullyConnectedLayerTest(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
    this.inputDim = inputDims;
    FullyConnectedLayer temp_11_0003 = new FullyConnectedLayer(inputDims, outputDims);
    temp_11_0003.setWeightsLog(-2);
    FullyConnectedLayer temp_11_0001 = temp_11_0003.addRef();
    temp_11_0003.freeRef();
    this.fullyConnectedLayer = temp_11_0001.addRef();
    temp_11_0001.freeRef();
    this.fullyConnectedLayer.setBatchBands(batchBands);
    FullyConnectedLayer temp_11_0004 = this.fullyConnectedLayer.addRef();
    Layer temp_11_0002 = temp_11_0004.explode();
    temp_11_0004.freeRef();
    this.layer = temp_11_0002.addRef();
    temp_11_0002.freeRef();
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return layer.addRef();
  }

  @Override
  public Layer getReferenceLayer() {
    @Nullable
    Class<? extends Layer> referenceLayerClass = getReferenceLayerClass();
    return null == referenceLayerClass ? null : this.fullyConnectedLayer.as(referenceLayerClass);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return FullyConnectedReferenceLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{inputDim};
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return FullyConnectedLayer.class;
  }

//  @Override
//  public void allTests(@Nonnull NotebookOutput log) {
//    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    //    log.p(log.file((String) null, logName, "GPU Log"));
//    //    CudaSystem.addLog(new PrintStream(log.file(logName)));
//    super.allTests(log);
//  }

  /**
   * Cleanup.
   */
  @AfterEach
  void cleanup() {
    if (null != layer)
      layer.freeRef();
    fullyConnectedLayer.freeRef();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(new int[]{2}, new int[]{2}, 512);
    }
  }

  /**
   * The type Big tests.
   */
  public abstract static class BigTests extends FullyConnectedLayerTest {

    /**
     * Instantiates a new Big tests.
     *
     * @param inputDims  the input dims
     * @param outputDims the output dims
     * @param batchBands the batch bands
     */
    public BigTests(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
      super(inputDims, outputDims, batchBands);
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, true, 5);
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

  /**
   * The type Big vgg.
   */
  public static class Big_VGG extends BigTests {
    /**
     * Instantiates a new Big vgg.
     */
    public Big_VGG() {
      super(new int[]{25088}, new int[]{4096}, 25088 / 2);
    }
  }

  /**
   * The type Big 1.
   */
  public static class Big1 extends BigTests {
    /**
     * Instantiates a new Big 1.
     */
    public Big1() {
      super(new int[]{2 * 1024}, new int[]{2 * 1024}, 512);
    }

  }

}
