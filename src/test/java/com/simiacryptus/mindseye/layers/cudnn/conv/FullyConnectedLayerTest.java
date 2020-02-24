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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.After;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class FullyConnectedLayerTest extends CudnnLayerTestBase {

  @Nonnull
  protected final int[] inputDim;
  @Nonnull
  @RefIgnore
  protected final FullyConnectedLayer fullyConnectedLayer;
  @Nonnull
  @RefIgnore
  protected final Layer layer;

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
  protected Class<?> getTargetClass() {
    return FullyConnectedLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{inputDim};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer.addRef();
  }

  @Override
  public void run(@Nonnull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }


  @After
  public void cleanup() {
    super.cleanup();
    if (null != layer)
      layer.freeRef();
    fullyConnectedLayer.freeRef();
  }

  public static class Basic extends FullyConnectedLayerTest {
    public Basic() {
      super(new int[]{2}, new int[]{2}, 512);
    }
  }

  public abstract static class BigTests extends FullyConnectedLayerTest {

    public BigTests(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
      super(inputDims, outputDims, batchBands);
      validateDifferentials = false;
      setTestTraining(false);
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution)
        return null;
      BatchingTester batchingTester = new BatchingTester(1e-2, true) {
        @Override
        public double getRandom() {
          return random();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
        }
      };
      batchingTester.setBatchSize(5);
      return batchingTester;
    }

    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
      //return super.getJsonTester();
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
      //return super.getPerformanceTester();
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

  }

  public static class Big_VGG extends BigTests {
    public Big_VGG() {
      super(new int[]{25088}, new int[]{4096}, 25088 / 2);
    }
  }

  public static class Big1 extends BigTests {
    public Big1() {
      super(new int[]{2 * 1024}, new int[]{2 * 1024}, 512);
    }

  }

}
