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
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Img tile subnet layer test.
 */
public abstract class ImgTileSubnetLayerTest extends CudnnLayerTestBase {

  @RefIgnore
  private final ConvolutionLayer convolutionLayer;

  {
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 1, 1);
    convolutionLayer.set(() -> this.random());
    this.convolutionLayer = convolutionLayer;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new ImgTileSubnetLayer(new ActivationLayer(ActivationLayer.Mode.RELU), 3, 3, 2, 2);
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return new ActivationLayer(ActivationLayer.Mode.RELU);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgTileSubnetLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{5, 5, 1}};
  }

  /**
   * Cleanup.
   */
  @AfterEach
  void cleanup() {
    if (null != convolutionLayer)
      convolutionLayer.freeRef();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends ImgTileSubnetLayerTest {

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }
  }

}
