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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.CudnnLayerTestBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.After;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public abstract class SimpleConvolutionLayerTest extends CudnnLayerTestBase {

  public final int radius;
  public final int bands;
  @Nullable
  @RefIgnore
  final SimpleConvolutionLayer layer;
  public int largeSize;
  public int smallSize;

  protected SimpleConvolutionLayerTest(final int radius, final int bands, final Precision precision, int stride) {
    this.radius = radius;
    this.bands = bands;
    SimpleConvolutionLayer simpleConvolutionLayer = new SimpleConvolutionLayer(radius, radius, bands * bands);
    simpleConvolutionLayer.setPrecision(precision);
    simpleConvolutionLayer.setStrideX(stride);
    simpleConvolutionLayer.setStrideY(stride);
    simpleConvolutionLayer.setWeightsLog(-2);
    layer = simpleConvolutionLayer;
    layer.set(() -> random());
    smallSize = this.radius;
    testTraining = false;
    largeSize = 800;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, bands}};
  }

  @Nullable
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, bands}};
  }

  @After
  public void cleanup() {
    super.cleanup();
    if (null != layer)
      layer.freeRef();
  }

  public static class Basic extends SimpleConvolutionLayerTest {
    public Basic() {
      super(1, 1, Precision.Double, 1);
    }

  }

  public static class Image extends SimpleConvolutionLayerTest {
    public Image() {
      super(3, 3, Precision.Double, 1);
      largeSize = 1200;
      smallSize = 5;
    }

  }

  public static class Image_Float extends SimpleConvolutionLayerTest {
    public Image_Float() {
      super(3, 3, Precision.Float, 1);
      tolerance = 1e-2;
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }

  public static class Matrix extends SimpleConvolutionLayerTest {
    public Matrix() {
      super(3, 1, Precision.Double, 1);
    }

  }

  public static class MultiBand extends SimpleConvolutionLayerTest {
    public MultiBand() {
      super(1, 3, Precision.Double, 1);
      smallSize = 8;
    }

  }

  public abstract static class Bug_Control extends SimpleConvolutionLayerTest {
    protected Bug_Control() {
      super(3, 8, Precision.Double, 1);
      validateDifferentials = false;
    }

    @Nonnull
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      PerformanceTester temp_10_0004 = new PerformanceTester();
      temp_10_0004.setBatches(10);
      PerformanceTester temp_10_0008 = temp_10_0004.addRef();
      temp_10_0008.setSamples(1);
      PerformanceTester temp_10_0003 = temp_10_0008.addRef();
      temp_10_0008.freeRef();
      temp_10_0004.freeRef();
      return temp_10_0003;
    }

    @Nonnull
    protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
      return new ReferenceIO(getReferenceIO());
    }

    @Override
    public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
      return null;
    }

    @Override
    @Test
    @Timeout(value = 15, unit = TimeUnit.MINUTES)
    public void test() throws Throwable {
      Assertions.assertThrows(Throwable.class, () -> {
        super.test();
      });
    }

    @Override
    public void run(@Nonnull NotebookOutput log) {
      //      @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
      //      log.p(log.file((String) null, logName, "GPU Log"));
      //      @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
      //      CudaSystem.addLog(apiLog);
      super.run(log);
      //      apiLog.close();
      //      CudaSystem.apiLog.remove(apiLog);
    }

  }

  public static class PaddingBug extends Image {
    public PaddingBug() {
      super();
      assert layer != null;
      layer.setPaddingXY(0, 0);
    }

  }

  public static class SpanBug extends Image {
    public SpanBug() {
      assert layer != null;
      layer.setStrideX(2);
      layer.setStrideY(2);
      largeSize = 800;
      smallSize = 5;
    }
  }

  public static class Big0 extends Big {
    public Big0() {
      super(1, 2048, Precision.Double);
    }

  }

  public abstract static class Big extends SimpleConvolutionLayerTest {
    public Big(int radius, int bands, Precision aDouble) {
      super(radius, bands, aDouble, 1);
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
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
    }

    @Override
    public Layer getReferenceLayer() {
      return null;
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{30, 30, bands}};
    }

  }

}
