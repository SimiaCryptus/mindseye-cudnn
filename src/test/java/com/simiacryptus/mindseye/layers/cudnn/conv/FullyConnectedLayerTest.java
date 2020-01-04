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
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class FullyConnectedLayerTest extends CudnnLayerTestBase {

  @Nonnull
  protected final int[] inputDim;
  @Nonnull
  protected final FullyConnectedLayer fullyConnectedLayer;
  @Nonnull
  protected final Layer layer;

  public FullyConnectedLayerTest(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
    this.inputDim = inputDims;
    this.fullyConnectedLayer = new FullyConnectedLayer(inputDims, outputDims).setWeightsLog(-2);
    this.layer = this.fullyConnectedLayer.setBatchBands(batchBands).explode();
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

  public static @SuppressWarnings("unused")
  FullyConnectedLayerTest[] addRefs(FullyConnectedLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayerTest::addRef)
        .toArray((x) -> new FullyConnectedLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  FullyConnectedLayerTest[][] addRefs(FullyConnectedLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayerTest::addRefs)
        .toArray((x) -> new FullyConnectedLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{inputDim};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer;
  }

  @Override
  public void run(@NotNull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  FullyConnectedLayerTest addRef() {
    return (FullyConnectedLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends FullyConnectedLayerTest {
    public Basic() {
      super(new int[]{2}, new int[]{2}, 512);
    }

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

  public abstract static @com.simiacryptus.ref.lang.RefAware
  class BigTests extends FullyConnectedLayerTest {

    public BigTests(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
      super(inputDims, outputDims, batchBands);
      validateDifferentials = false;
      setTestTraining(false);
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution)
        return null;
      return (new BatchingTester(1e-2, true) {
        @Override
        public double getRandom() {
          return random();
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      }).setBatchSize(5);
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

    public static @SuppressWarnings("unused")
    BigTests[] addRefs(BigTests[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BigTests::addRef)
          .toArray((x) -> new BigTests[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    BigTests addRef() {
      return (BigTests) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Big_VGG extends BigTests {
    public Big_VGG() {
      super(new int[]{25088}, new int[]{4096}, 25088 / 2);
    }

    public static @SuppressWarnings("unused")
    Big_VGG[] addRefs(Big_VGG[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Big_VGG::addRef)
          .toArray((x) -> new Big_VGG[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Big_VGG addRef() {
      return (Big_VGG) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Big1 extends BigTests {
    public Big1() {
      super(new int[]{2 * 1024}, new int[]{2 * 1024}, 512);
    }

    public static @SuppressWarnings("unused")
    Big1[] addRefs(Big1[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Big1::addRef).toArray((x) -> new Big1[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Big1 addRef() {
      return (Big1) super.addRef();
    }

  }

}
