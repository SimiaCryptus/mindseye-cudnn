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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.mindseye.test.unit.TrainingTester;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class ImgBandDynamicBiasLayerTest extends CudnnLayerTestBase {

  final Precision precision;

  public ImgBandDynamicBiasLayerTest(final Precision precision) {
    this.precision = precision;
    this.testingBatchSize = 1;
    this.validateBatchExecution = false;

  }

  @Nullable
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return isTestTraining() ? new TrainingTester() {
      @Override
      protected Layer lossLayer() {
        return new MeanSqLossLayer();
      }
    }.setBatches(1) : null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {8, 8, 3}, {1, 1, 3}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgBandDynamicBiasLayer().setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
        {1200, 1200, 3}, {1, 1, 3}
    };
  }

  public static class Double extends ImgBandDynamicBiasLayerTest {
    public Double() {
      super(Precision.Double);
    }
  }

  public static class Float extends ImgBandDynamicBiasLayerTest {
    public Float() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}