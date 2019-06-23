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
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class SimpleConvolutionLayerTest extends CudnnLayerTestBase {

  public final int radius;
  public final int bands;
  public int largeSize;
  public int smallSize;
  SimpleConvolutionLayer layer;


  protected SimpleConvolutionLayerTest(final int radius, final int bands, final Precision precision, int stride) {
    this.radius = radius;
    this.bands = bands;
    layer = new SimpleConvolutionLayer(radius, radius, bands * bands).setPrecision(precision).setStrideX(stride).setStrideY(stride).setWeightsLog(-2);
    layer.kernel.set(() -> random());
    smallSize = this.radius;
    testTraining = false;
    largeSize = 800;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {smallSize, smallSize, bands}
    };
  }

  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    layer.addRef();
    return layer;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
        {largeSize, largeSize, bands}
    };
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return null;
//    @Nonnull final ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, bands, bands, true);
//    @Nonnull final Tensor tensor = new Tensor(layer.kernel.getDimensions());
//    tensor.setByCoord(c -> {
//      final int band = c.getCoords()[2];
//      final int bandX = band % bands;
//      final int bandY = (band - bandX) / bands;
//      assert band == bandX + bandY * bands;
//      final int bandT = bandY + bandX * bands;
//      return layer.kernel.get(c.getCoords()[0], c.getCoords()[1], bandT);
//    });
//    convolutionLayer.kernel.set(tensor);
//    tensor.freeRef();
//    return convolutionLayer;
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

    @Override
    @Test(timeout = 15 * 60 * 1000, expected = Throwable.class)
    public void test() throws Throwable {
      super.test();
    }

    @Override
    public void run(NotebookOutput log) {
//      @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//      log.p(log.file((String) null, logName, "GPU Log"));
//      @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
//      CudaSystem.addLog(apiLog);
      super.run(log);
//      apiLog.close();
//      CudaSystem.apiLog.remove(apiLog);
    }


    @Override
    public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
      return null;
    }

    @Nonnull
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      return new PerformanceTester().setBatches(10).setSamples(1);
    }

    @Nonnull
    protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
      return new ReferenceIO(getReferenceIO());
    }

  }

  public static class PaddingBug extends Image {
    public PaddingBug() {
      super();
      layer.setPaddingXY(0, 0);
    }

  }

  public static class SpanBug extends Image {
    public SpanBug() {
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
    public Layer getReferenceLayer() {
      return null;
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
          {30, 30, bands}
      };
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution) return null;
      return (new BatchingTester(1e-2, true) {
        @Override
        public double getRandom() {
          return random();
        }
      }).setBatchSize(5);
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

  }

}
