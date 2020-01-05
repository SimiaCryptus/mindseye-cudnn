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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class SimpleConvolutionLayerTest extends CudnnLayerTestBase {

  public final int radius;
  public final int bands;
  final SimpleConvolutionLayer layer;
  public int largeSize;
  public int smallSize;

  protected SimpleConvolutionLayerTest(final int radius, final int bands, final Precision precision, int stride) {
    this.radius = radius;
    this.bands = bands;
    {
      SimpleConvolutionLayer temp_10_0002 = new SimpleConvolutionLayer(
          radius, radius, bands * bands);
      SimpleConvolutionLayer temp_10_0005 = temp_10_0002
          .setPrecision(precision);
      SimpleConvolutionLayer temp_10_0006 = temp_10_0005.setStrideX(stride);
      SimpleConvolutionLayer temp_10_0007 = temp_10_0006.setStrideY(stride);
      SimpleConvolutionLayer temp_10_0001 = temp_10_0007.setWeightsLog(-2);
      if (null != temp_10_0007)
        temp_10_0007.freeRef();
      if (null != temp_10_0006)
        temp_10_0006.freeRef();
      if (null != temp_10_0005)
        temp_10_0005.freeRef();
      if (null != temp_10_0002)
        temp_10_0002.freeRef();
      layer = temp_10_0001 == null ? null : temp_10_0001.addRef();
      if (null != temp_10_0001)
        temp_10_0001.freeRef();
    }
    RefUtil.freeRef(layer.kernel.set(() -> random()));
    smallSize = this.radius;
    testTraining = false;
    largeSize = 800;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return null;
  }

  public static @SuppressWarnings("unused")
  SimpleConvolutionLayerTest[] addRefs(SimpleConvolutionLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleConvolutionLayerTest::addRef)
        .toArray((x) -> new SimpleConvolutionLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  SimpleConvolutionLayerTest[][] addRefs(
      SimpleConvolutionLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleConvolutionLayerTest::addRefs)
        .toArray((x) -> new SimpleConvolutionLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, bands}};
  }

  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, bands}};
  }

  public @SuppressWarnings("unused")
  void _free() {
    if (null != layer)
      layer.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleConvolutionLayerTest addRef() {
    return (SimpleConvolutionLayerTest) super.addRef();
  }

  public static @RefAware
  class Basic extends SimpleConvolutionLayerTest {
    public Basic() {
      super(1, 1, Precision.Double, 1);
    }

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
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

  public static @RefAware
  class Image extends SimpleConvolutionLayerTest {
    public Image() {
      super(3, 3, Precision.Double, 1);
      largeSize = 1200;
      smallSize = 5;
    }

    public static @SuppressWarnings("unused")
    Image[] addRefs(Image[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Image::addRef).toArray((x) -> new Image[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Image addRef() {
      return (Image) super.addRef();
    }
  }

  public static @RefAware
  class Image_Float extends SimpleConvolutionLayerTest {
    public Image_Float() {
      super(3, 3, Precision.Float, 1);
      tolerance = 1e-2;
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    public static @SuppressWarnings("unused")
    Image_Float[] addRefs(Image_Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Image_Float::addRef)
          .toArray((x) -> new Image_Float[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Image_Float addRef() {
      return (Image_Float) super.addRef();
    }

  }

  public static @RefAware
  class Matrix extends SimpleConvolutionLayerTest {
    public Matrix() {
      super(3, 1, Precision.Double, 1);
    }

    public static @SuppressWarnings("unused")
    Matrix[] addRefs(Matrix[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Matrix::addRef).toArray((x) -> new Matrix[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Matrix addRef() {
      return (Matrix) super.addRef();
    }
  }

  public static @RefAware
  class MultiBand extends SimpleConvolutionLayerTest {
    public MultiBand() {
      super(1, 3, Precision.Double, 1);
      smallSize = 8;
    }

    public static @SuppressWarnings("unused")
    MultiBand[] addRefs(MultiBand[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(MultiBand::addRef)
          .toArray((x) -> new MultiBand[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    MultiBand addRef() {
      return (MultiBand) super.addRef();
    }
  }

  public abstract static @RefAware
  class Bug_Control extends SimpleConvolutionLayerTest {
    protected Bug_Control() {
      super(3, 8, Precision.Double, 1);
      validateDifferentials = false;
    }

    @Nonnull
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      PerformanceTester temp_10_0004 = new PerformanceTester();
      PerformanceTester temp_10_0008 = temp_10_0004.setBatches(10);
      PerformanceTester temp_10_0003 = temp_10_0008.setSamples(1);
      if (null != temp_10_0008)
        temp_10_0008.freeRef();
      if (null != temp_10_0004)
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

    public static @SuppressWarnings("unused")
    Bug_Control[] addRefs(Bug_Control[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Bug_Control::addRef)
          .toArray((x) -> new Bug_Control[x]);
    }

    @Override
    @Test(timeout = 15 * 60 * 1000, expected = Throwable.class)
    public void test() throws Throwable {
      super.test();
    }

    @Override
    public void run(@NotNull NotebookOutput log) {
      //      @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
      //      log.p(log.file((String) null, logName, "GPU Log"));
      //      @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
      //      CudaSystem.addLog(apiLog);
      super.run(log);
      //      apiLog.close();
      //      CudaSystem.apiLog.remove(apiLog);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Bug_Control addRef() {
      return (Bug_Control) super.addRef();
    }

  }

  public static @RefAware
  class PaddingBug extends Image {
    public PaddingBug() {
      super();
      layer.setPaddingXY(0, 0);
    }

    public static @SuppressWarnings("unused")
    PaddingBug[] addRefs(PaddingBug[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(PaddingBug::addRef)
          .toArray((x) -> new PaddingBug[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    PaddingBug addRef() {
      return (PaddingBug) super.addRef();
    }

  }

  public static @RefAware
  class SpanBug extends Image {
    public SpanBug() {
      RefUtil.freeRef(layer.setStrideX(2));
      RefUtil.freeRef(layer.setStrideY(2));
      largeSize = 800;
      smallSize = 5;
    }

    public static @SuppressWarnings("unused")
    SpanBug[] addRefs(SpanBug[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(SpanBug::addRef)
          .toArray((x) -> new SpanBug[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    SpanBug addRef() {
      return (SpanBug) super.addRef();
    }
  }

  public static @RefAware
  class Big0 extends Big {
    public Big0() {
      super(1, 2048, Precision.Double);
    }

    public static @SuppressWarnings("unused")
    Big0[] addRefs(Big0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big0::addRef).toArray((x) -> new Big0[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Big0 addRef() {
      return (Big0) super.addRef();
    }
  }

  public abstract static @RefAware
  class Big extends SimpleConvolutionLayerTest {
    public Big(int radius, int bands, Precision aDouble) {
      super(radius, bands, aDouble, 1);
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

    public static @SuppressWarnings("unused")
    Big[] addRefs(Big[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big::addRef).toArray((x) -> new Big[x]);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{{30, 30, bands}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Big addRef() {
      return (Big) super.addRef();
    }

  }

}
