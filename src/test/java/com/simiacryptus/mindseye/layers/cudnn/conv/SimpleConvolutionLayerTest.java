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
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

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
    largeSize = 800;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, bands}};
  }

  @Nullable
  @Override
  public Layer getLayer() {
    return layer == null ? null : layer.addRef();
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, bands}};
  }

  @Override
  @Disabled
  public void trainingTest() {
    super.trainingTest();
  }

  @AfterEach
  void cleanup() {
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
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, false, this.testingBatchSize);
    }

    public @Nullable PerformanceTester getPerformanceTester() {
      PerformanceTester performanceTester = new PerformanceTester();
      performanceTester.setBatches(10);
      performanceTester.setSamples(1);
      return performanceTester;
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

//    @Override
//    public void allTests(@Nonnull NotebookOutput log) {
//      //      @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//      //      log.p(log.file((String) null, logName, "GPU Log"));
//      //      @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
//      //      CudaSystem.addLog(apiLog);
//      super.allTests(log);
//      //      apiLog.close();
//      //      CudaSystem.apiLog.remove(apiLog);
//    }

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
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, true, 5);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{30, 30, bands}};
    }

    @Override
    public Layer getReferenceLayer() {
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
