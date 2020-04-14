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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;

/**
 * The type Img modulus padding layer test.
 */
public abstract class ImgModulusPaddingLayerTest extends CudnnLayerTestBase {

  /**
   * The Modulus.
   */
  final int modulus;
  /**
   * The Offset.
   */
  final int offset;

  /**
   * Instantiates a new Img modulus padding layer test.
   *
   * @param inputSize the input size
   * @param modulus   the modulus
   * @param offset    the offset
   */
  public ImgModulusPaddingLayerTest(int inputSize, int modulus, int offset) {
    this.modulus = modulus;
    this.offset = offset;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new ImgModulusPaddingLayer(modulus, modulus, offset, offset);
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2, 1}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends ImgModulusPaddingLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(2, 3, 0);
    }
  }

}
