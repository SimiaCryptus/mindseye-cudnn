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
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefAssert;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class ConvolutionLayerTest extends CudnnLayerTestBase {

  final int inputBands;
  final int outputBands;
  final int radius;
  @Nullable
  @RefIgnore
  final ConvolutionLayer convolutionLayer;
  final int smallSize;
  final int largeSize;

  protected ConvolutionLayerTest(final int radius, final int inputBands, final int outputBands,
                                 final Precision precision, int batchBands, int stride, final int smallSize, final int largeSize) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands);
    convolutionLayer.setPrecision(precision);
    convolutionLayer.setBatchBands(batchBands);
    convolutionLayer.setStrideXY(stride, stride);
    @Nonnull
    Random random = getRandom();
    convolutionLayer.set(() -> {
      return random(random);
    });
    this.convolutionLayer = convolutionLayer;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
    this.testingBatchSize = 2;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, inputBands}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    assert convolutionLayer != null;
    return convolutionLayer.explode();
  }

//  @Override
//  public void allTests(@Nonnull NotebookOutput log) {
//    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    //    log.p(log.file((String) null, logName, "GPU Log"));
//    //    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
//    //    CudaSystem.addLog(apiLog);
//    super.allTests(log);
//    //    apiLog.close();
//    //    CudaSystem.apiLog.remove(apiLog);
//  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    assert convolutionLayer != null;
    //return convolutionLayer.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, inputBands}};
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return ConvolutionLayer.class;
  }

  @Test
  public void verifyWeights() {
    assert this.convolutionLayer != null;
    @Nonnull
    ExplodedConvolutionGrid explodedNetwork = this.convolutionLayer.getExplodedNetwork();
    @Nonnull
    int[] kernelDims = this.convolutionLayer.getKernelDimensions();
    Tensor testData = new Tensor(kernelDims);
    testData.set(i->random());
    explodedNetwork.write(testData.addRef());
    Tensor echo = explodedNetwork.read();
    explodedNetwork.freeRef();
    if (!testData.equals(echo)) {
      Tensor minus = testData.minus(echo == null ? null : echo.addRef());
      print(minus.coordStream(false)
          .filter(x -> minus.get(x) != 0)
          .map(x -> String.format("%s=%s", x, minus.get(x))));
      minus.freeRef();
      RefAssert.assertEquals(testData, echo);
    } else {
      if (null != echo)
        echo.freeRef();
      testData.freeRef();
    }
  }

  @AfterEach
  void cleanup() {
    if (null != convolutionLayer)
      convolutionLayer.freeRef();
  }

  private void print(@Nonnull final RefStream<CharSequence> stream) {
    stream.forEach(x -> System.out.println("Zero: " + x));
    //com.simiacryptus.ref.wrappers.System.out.println("Zeros: " + stream.sumChannels((a,b)->a+","+b).get());
  }

  public static class BandExpand extends ConvolutionLayerTest {

    public BandExpand() {
      super(1, 3, 6, Precision.Double, 16, 1, 3, 600);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return getSmallDims();
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{1, 1, inputBands}};
    }

  }

  public static class BandLimit extends ConvolutionLayerTest {

    public BandLimit() {
      super(1, 3, 2, Precision.Double, 16, 1, 3, 600);
    }
  }

  public static class SqGrid extends ConvolutionLayerTest {

    public SqGrid() {
      super(3, 4, 4, Precision.Double, 2, 1, 3, 600);
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

  public static class IrregularGrid extends ConvolutionLayerTest {

    public IrregularGrid() {
      super(3, 5, 3, Precision.Double, 2, 1, 3, 600);
    }
  }

  public static class BandReduceTest extends ConvolutionLayerTest {

    public BandReduceTest() {
      super(3, 6, 3, Precision.Double, 16, 1, 3, 600);
    }
  }

  public static class Double extends ConvolutionLayerTest {

    public Double() {
      super(3, 4, 4, Precision.Double, 16, 1, 3, 600);
    }
  }

  public static class NoPadding extends ConvolutionLayerTest {
    public NoPadding() {
      super(3, 3, 3, Precision.Double, 16, 1, 3, 600);
      assert convolutionLayer != null;
      convolutionLayer.setPaddingXY(0, 0);
      RefUtil.freeRef(convolutionLayer.addRef());
    }

  }

  public static class Float extends ConvolutionLayerTest {
    public Float() {
      super(1, 2, 2, Precision.Float, 16, 1, 3, 600);
    }
  }

  public static class IrregularTest extends ConvolutionLayerTest {

    public IrregularTest() {
      super(3, 7, 5, Precision.Double, 16, 1, 3, 600);
    }

  }

  public static class IrregularTest_Float extends ConvolutionLayerTest {

    public IrregularTest_Float() {
      super(3, 7, 5, Precision.Float, 16, 1, 3, 600);
    }

    @Override
    public @Nullable SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-1, 1e-4);
    }

  }

  public static class Big1 extends VeryBigTest {
    public Big1() {
      this(1024);
    }

    private Big1(int size) {
      super(1, size, size, Precision.Float, size);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{256, 128, inputBands}};
    }

  }

  public abstract static class VeryBigTest extends Big {

    protected VeryBigTest(final int radius, final int inputBands, final int outputBands, final Precision precision,
                          final int batchBands) {
      super(radius, inputBands, outputBands, precision, batchBands);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{100, 100, inputBands}};
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{1, 1, inputBands}};
    }

  }

  public abstract static class Big extends ConvolutionLayerTest {

    public Big(final int radius, final int inputBands, final int outputBands, final Precision precision,
               int batchBands) {
      super(radius, inputBands, outputBands, precision, batchBands, 1, 3, 600);
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, true, 5);
    }

    @Nullable
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
