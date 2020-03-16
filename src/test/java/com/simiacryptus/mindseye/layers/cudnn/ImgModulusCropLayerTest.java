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
import org.junit.Ignore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.TestInfo;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract class ImgModulusCropLayerTest extends CudnnLayerTestBase {

  final int modulus;
  final int offset;

  public ImgModulusCropLayerTest(int modulus, int offset) {
    testingBatchSize = 5;
    this.modulus = modulus;
    this.offset = offset;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Override
  @Disabled
  public void derivativeTest(TestInfo testInfo) {
    super.derivativeTest(testInfo);
  }

  @Override
  @Disabled
  public void batchingTest(TestInfo testInfo) {
    super.batchingTest(testInfo);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{320, 240, 3}
        //        {2, 2, 1}
    };
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{320, 240, 3}
        //        {1200, 1200, 1}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgModulusCropLayer(modulus, modulus, offset, offset)
        //.setRoundUp(true)
        ;
  }

  public static class Basic extends ImgModulusCropLayerTest {
    public Basic() {
      super(2, 7);
    }

  }

}
