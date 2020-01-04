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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.CudnnLayerTestBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.notebook.NotebookOutput;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class ConvolutionLayerTest extends CudnnLayerTestBase {

  final int inputBands;
  final int outputBands;
  final int radius;
  final ConvolutionLayer convolutionLayer;
  final int smallSize;
  final int largeSize;

  protected ConvolutionLayerTest(final int radius, final int inputBands, final int outputBands,
                                 final Precision precision, int batchBands, int stride, final int smallSize, final int largeSize) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision)
        .setBatchBands(batchBands).setStrideXY(stride, stride);
    @Nonnull
    Random random = getRandom();
    convolutionLayer.getKernel().set(() -> {
      return random(random);
    });
    this.smallSize = smallSize;
    this.largeSize = largeSize;
    this.testingBatchSize = 2;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return convolutionLayer.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return ConvolutionLayer.class;
  }

  public static @SuppressWarnings("unused")
  ConvolutionLayerTest[] addRefs(ConvolutionLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayerTest::addRef)
        .toArray((x) -> new ConvolutionLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ConvolutionLayerTest[][] addRefs(ConvolutionLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayerTest::addRefs)
        .toArray((x) -> new ConvolutionLayerTest[x][]);
  }

  @Override
  public void run(@NotNull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    //    CudaSystem.addLog(apiLog);
    super.run(log);
    //    apiLog.close();
    //    CudaSystem.apiLog.remove(apiLog);
  }

  @Test
  public void verifyWeights() {
    @Nonnull
    ExplodedConvolutionGrid explodedNetwork = this.convolutionLayer.getExplodedNetwork();
    @Nonnull
    int[] kernelDims = this.convolutionLayer.getKernel().getDimensions();
    @Nullable
    Tensor testData = new Tensor(kernelDims).map(x -> random());
    explodedNetwork.write(testData);
    Tensor echo = explodedNetwork.read();
    if (!testData.equals(echo)) {
      Tensor minus = testData.minus(echo);
      print(minus.coordStream(false).filter(x -> minus.get(x) != 0).map(x -> String.format("%s=%s", x, minus.get(x))));
      com.simiacryptus.ref.wrappers.RefAssert.assertEquals(testData, echo);
    }
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, inputBands}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return convolutionLayer.explode();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, inputBands}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ConvolutionLayerTest addRef() {
    return (ConvolutionLayerTest) super.addRef();
  }

  private void print(final com.simiacryptus.ref.wrappers.RefStream<CharSequence> stream) {
    stream.forEach(x -> System.out.println("Zero: " + x));
    //System.out.println("Zeros: " + stream.sumChannels((a,b)->a+","+b).get());
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class BandExpand extends ConvolutionLayerTest {

    public BandExpand() {
      super(1, 3, 6, Precision.Double, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    BandExpand[] addRefs(BandExpand[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandExpand::addRef)
          .toArray((x) -> new BandExpand[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{1, 1, inputBands}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(random);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    BandExpand addRef() {
      return (BandExpand) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class BandLimit extends ConvolutionLayerTest {

    public BandLimit() {
      super(1, 3, 2, Precision.Double, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    BandLimit[] addRefs(BandLimit[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandLimit::addRef)
          .toArray((x) -> new BandLimit[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    BandLimit addRef() {
      return (BandLimit) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class SqGrid extends ConvolutionLayerTest {

    public SqGrid() {
      super(3, 4, 4, Precision.Double, 2, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    SqGrid[] addRefs(SqGrid[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(SqGrid::addRef).toArray((x) -> new SqGrid[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    SqGrid addRef() {
      return (SqGrid) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class IrregularGrid extends ConvolutionLayerTest {

    public IrregularGrid() {
      super(3, 5, 3, Precision.Double, 2, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    IrregularGrid[] addRefs(IrregularGrid[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(IrregularGrid::addRef)
          .toArray((x) -> new IrregularGrid[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    IrregularGrid addRef() {
      return (IrregularGrid) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class BandReduceTest extends ConvolutionLayerTest {

    public BandReduceTest() {
      super(3, 6, 3, Precision.Double, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    BandReduceTest[] addRefs(BandReduceTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(BandReduceTest::addRef)
          .toArray((x) -> new BandReduceTest[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    BandReduceTest addRef() {
      return (BandReduceTest) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Double extends ConvolutionLayerTest {

    public Double() {
      super(3, 4, 4, Precision.Double, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Double addRef() {
      return (Double) super.addRef();
    }

  }

  //  /**
  //   * The type BigTests temp 0.
  //   */
  //  public static class Big0 extends VeryBigTest {
  //    /**
  //     * Instantiates a new BigTests.
  //     */
  //    public Big0() {this(512);}
  //
  //    /**
  //     * Instantiates a new BigTests.
  //     *
  //     * @param size
  //     */
  //    private Big0(int size) {
  //      super(1, 16 * size, 16 * size, Precision.Double, size);
  //    }
  //
  //  }

  public static @com.simiacryptus.ref.lang.RefAware
  class NoPadding extends ConvolutionLayerTest {
    public NoPadding() {
      super(3, 3, 3, Precision.Double, 16, 1, 3, 600);
      convolutionLayer.setPaddingXY(0, 0);
    }

    public static @SuppressWarnings("unused")
    NoPadding[] addRefs(NoPadding[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(NoPadding::addRef)
          .toArray((x) -> new NoPadding[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    NoPadding addRef() {
      return (NoPadding) super.addRef();
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Float extends ConvolutionLayerTest {
    public Float() {
      super(1, 2, 2, Precision.Float, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Float addRef() {
      return (Float) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class IrregularTest extends ConvolutionLayerTest {

    public IrregularTest() {
      super(3, 7, 5, Precision.Double, 16, 1, 3, 600);
    }

    public static @SuppressWarnings("unused")
    IrregularTest[] addRefs(IrregularTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(IrregularTest::addRef)
          .toArray((x) -> new IrregularTest[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    IrregularTest addRef() {
      return (IrregularTest) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class IrregularTest_Float extends ConvolutionLayerTest {

    public IrregularTest_Float() {
      super(3, 7, 5, Precision.Float, 16, 1, 3, 600);
    }

    @Override
    public ComponentTest<ToleranceStatistics> getDerivativeTester() {
      if (!validateDifferentials)
        return null;
      return new SingleDerivativeTester(1e-1, 1e-4);
    }

    public static @SuppressWarnings("unused")
    IrregularTest_Float[] addRefs(IrregularTest_Float[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(IrregularTest_Float::addRef)
          .toArray((x) -> new IrregularTest_Float[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    IrregularTest_Float addRef() {
      return (IrregularTest_Float) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Big1 extends VeryBigTest {
    public Big1() {
      this(1024);
    }

    private Big1(int size) {
      super(1, size, size, Precision.Float, size);
    }

    public static @SuppressWarnings("unused")
    Big1[] addRefs(Big1[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Big1::addRef).toArray((x) -> new Big1[x]);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{256, 128, inputBands}};
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

  public abstract static @com.simiacryptus.ref.lang.RefAware
  class VeryBigTest extends Big {

    protected VeryBigTest(final int radius, final int inputBands, final int outputBands, final Precision precision,
                          final int batchBands) {
      super(radius, inputBands, outputBands, precision, batchBands);
    }

    public static @SuppressWarnings("unused")
    VeryBigTest[] addRefs(VeryBigTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(VeryBigTest::addRef)
          .toArray((x) -> new VeryBigTest[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(final Random random) {
      return new int[][]{{1, 1, inputBands}};
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{100, 100, inputBands}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    VeryBigTest addRef() {
      return (VeryBigTest) super.addRef();
    }
  }

  public abstract static @com.simiacryptus.ref.lang.RefAware
  class Big extends ConvolutionLayerTest {

    public Big(final int radius, final int inputBands, final int outputBands, final Precision precision,
               int batchBands) {
      super(radius, inputBands, outputBands, precision, batchBands, 1, 3, 600);
      validateDifferentials = false;
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

    @Nullable
    @Override
    public Layer getReferenceLayer() {
      return null;
    }

    public static @SuppressWarnings("unused")
    Big[] addRefs(Big[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Big::addRef).toArray((x) -> new Big[x]);
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
