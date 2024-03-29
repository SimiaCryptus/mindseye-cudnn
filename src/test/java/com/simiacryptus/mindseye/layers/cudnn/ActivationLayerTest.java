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

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import smile.plot.swing.PlotPanel;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleFunction;

/**
 * The type Activation layer test.
 */
public abstract class ActivationLayerTest extends CudnnLayerTestBase {

  /**
   * The Mode.
   */
  final ActivationLayer.Mode mode;
  private final Precision precision;
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Activation layer test.
   *
   * @param mode      the mode
   * @param precision the precision
   * @param smallSize the small size
   * @param largeSize the large size
   */
  public ActivationLayerTest(final ActivationLayer.Mode mode, final Precision precision, final int smallSize,
                             final int largeSize) {
    this.mode = mode;
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Override
  public SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(1e-2, 1e-4);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    ActivationLayer activationLayer = new ActivationLayer(mode);
    activationLayer.setPrecision(precision);
    return activationLayer;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, 1}};
  }

  /**
   * Function plots.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void functionPlots() {
    NotebookOutput log = getLog();
    log.h3("Function Plots");
    @Nonnull final Layer layer = getLayer();
    final RefList<double[]> plotData = RefIntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0)
        .mapToObj(RefUtil.wrapInterface((DoubleFunction<? extends double[]>) x -> {
          @Nonnull
          Tensor input = new Tensor(new double[]{x}, 1, 1, 1);
          @Nonnull final SimpleEval eval = SimpleEval.run(layer.addRef(), input);
          Tensor output = eval.getOutput();
          Tensor[] derivative = eval.getDerivative();
          assert derivative != null;
          assert output != null;
          double[] doubles = new double[]{x, output.get(0), derivative[0].get(0)};
          RefUtil.freeRef(derivative);
          output.freeRef();
          eval.freeRef();
          return doubles;
        }, layer)).collect(RefCollectors.toList());
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotPanel>) () -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[1]});
    }, plotData == null ? null : plotData.addRef()));

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotPanel>) () -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[2]});
    }, plotData == null ? null : plotData.addRef()));
    if (null != plotData)
      plotData.freeRef();
  }

  //  @Override
//  public void allTests(@Nonnull final NotebookOutput log) {
//    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    //    log.p(log.file((String) null, logName, "GPU Log"));
//    //    CudaSystem.addLog(new PrintStream(log.file(logName)));
//    super.allTests(log);
//  }

  /**
   * The type Re lu double.
   */
  public static class ReLu_Double extends ActivationLayerTest {
    /**
     * Instantiates a new Re lu double.
     */
    public ReLu_Double() {
      super(ActivationLayer.Mode.RELU, Precision.Double, 2, 800);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }

  }

  /**
   * The type Re lu float.
   */
  public static class ReLu_Float extends ActivationLayerTest {
    /**
     * Instantiates a new Re lu float.
     */
    public ReLu_Float() {
      super(ActivationLayer.Mode.RELU, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }

  }

  /**
   * The type Sigmoid double.
   */
  public static class Sigmoid_Double extends ActivationLayerTest {
    /**
     * Instantiates a new Sigmoid double.
     */
    public Sigmoid_Double() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Double, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      SigmoidActivationLayer temp_21_0005 = new SigmoidActivationLayer();
      temp_21_0005.setBalanced(false);
      SigmoidActivationLayer temp_21_0004 = temp_21_0005.addRef();
      temp_21_0005.freeRef();
      return temp_21_0004;
    }

  }

  /**
   * The type Sigmoid float.
   */
  public static class Sigmoid_Float extends ActivationLayerTest {
    /**
     * Instantiates a new Sigmoid float.
     */
    public Sigmoid_Float() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      SigmoidActivationLayer temp_21_0007 = new SigmoidActivationLayer();
      temp_21_0007.setBalanced(false);
      SigmoidActivationLayer temp_21_0006 = temp_21_0007.addRef();
      temp_21_0007.freeRef();
      return temp_21_0006;
    }

  }

}
