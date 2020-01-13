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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import smile.plot.PlotCanvas;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

public abstract class ActivationLayerTest extends CudnnLayerTestBase {

  final ActivationLayer.Mode mode;
  private final Precision precision;
  private final int smallSize;
  private final int largeSize;

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

  public static @SuppressWarnings("unused") ActivationLayerTest[] addRefs(ActivationLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayerTest::addRef)
        .toArray((x) -> new ActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ActivationLayerTest[][] addRefs(ActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayerTest::addRefs)
        .toArray((x) -> new ActivationLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { smallSize, smallSize, 1 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    ActivationLayer temp_21_0003 = new ActivationLayer(mode);
    ActivationLayer temp_21_0002 = temp_21_0003.setPrecision(precision);
    if (null != temp_21_0003)
      temp_21_0003.freeRef();
    return temp_21_0002;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { largeSize, largeSize, 1 } };
  }

  @Override
  public void run(@NotNull final NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    CudaSystem.addLog(new PrintStream(log.file(logName)));

    super.run(log);

    log.h3("Function Plots");
    @Nonnull
    final Layer layer = getLayer(new int[][] { { 1, 1, 1 } }, new Random());
    final RefList<double[]> plotData = RefIntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0)
        .mapToObj(RefUtil.wrapInterface((DoubleFunction<? extends double[]>) x -> {
          @Nonnull
          Tensor input = new Tensor(new double[] { x }, 1, 1, 1);
          @Nonnull
          final SimpleEval eval = SimpleEval.run(layer == null ? null : layer.addRef(), input == null ? null : input);
          Tensor temp_21_0008 = eval.getOutput();
          double[] temp_21_0001 = new double[] { x, temp_21_0008.get(0), eval.getDerivative()[0].get(0) };
          if (null != temp_21_0008)
            temp_21_0008.freeRef();
          eval.freeRef();
          return temp_21_0001;
        }, layer == null ? null : layer)).collect(RefCollectors.toList());
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[] { x[0], x[1] });
    }, plotData == null ? null : plotData.addRef()));

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[] { x[0], x[2] });
    }, plotData == null ? null : plotData.addRef()));
    if (null != plotData)
      plotData.freeRef();

  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ActivationLayerTest addRef() {
    return (ActivationLayerTest) super.addRef();
  }

  public static class ReLu_Double extends ActivationLayerTest {
    public ReLu_Double() {
      super(ActivationLayer.Mode.RELU, Precision.Double, 2, 800);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }

    public static @SuppressWarnings("unused") ReLu_Double[] addRefs(ReLu_Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(ReLu_Double::addRef).toArray((x) -> new ReLu_Double[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") ReLu_Double addRef() {
      return (ReLu_Double) super.addRef();
    }
  }

  public static class ReLu_Float extends ActivationLayerTest {
    public ReLu_Float() {
      super(ActivationLayer.Mode.RELU, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }

    public static @SuppressWarnings("unused") ReLu_Float[] addRefs(ReLu_Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(ReLu_Float::addRef).toArray((x) -> new ReLu_Float[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") ReLu_Float addRef() {
      return (ReLu_Float) super.addRef();
    }
  }

  public static class Sigmoid_Double extends ActivationLayerTest {
    public Sigmoid_Double() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Double, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      SigmoidActivationLayer temp_21_0005 = new SigmoidActivationLayer();
      SigmoidActivationLayer temp_21_0004 = temp_21_0005.setBalanced(false);
      if (null != temp_21_0005)
        temp_21_0005.freeRef();
      return temp_21_0004;
    }

    public static @SuppressWarnings("unused") Sigmoid_Double[] addRefs(Sigmoid_Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Sigmoid_Double::addRef)
          .toArray((x) -> new Sigmoid_Double[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Sigmoid_Double addRef() {
      return (Sigmoid_Double) super.addRef();
    }
  }

  public static class Sigmoid_Float extends ActivationLayerTest {
    public Sigmoid_Float() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      SigmoidActivationLayer temp_21_0007 = new SigmoidActivationLayer();
      SigmoidActivationLayer temp_21_0006 = temp_21_0007.setBalanced(false);
      if (null != temp_21_0007)
        temp_21_0007.freeRef();
      return temp_21_0006;
    }

    public static @SuppressWarnings("unused") Sigmoid_Float[] addRefs(Sigmoid_Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Sigmoid_Float::addRef)
          .toArray((x) -> new Sigmoid_Float[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Sigmoid_Float addRef() {
      return (Sigmoid_Float) super.addRef();
    }
  }

}
