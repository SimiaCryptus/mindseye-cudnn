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
import com.simiacryptus.ref.lang.RefUtil;
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Img tile select layer test.
 */
public abstract class ImgTileSelectLayerTest extends CudnnLayerTestBase {

  /**
   * Instantiates a new Img tile select layer test.
   */
  public ImgTileSelectLayerTest() {
  }

  @Nonnull
  @Override
  public abstract ImgTileSelectLayer getLayer();

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer.class;
  }

  //  @Override
  //  public int[][] getLargeDims(final Random random) {
  //    return new int[][]{
  //        {1200, 1200, 1}
  //    };
  //  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 6, 1}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type Ul.
   */
  public static class UL extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer() {
      ImgTileSelectLayer temp_58_0002 = new ImgTileSelectLayer(4, 3, 0, 0);
      temp_58_0002.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0001 = RefUtil.addRef(temp_58_0002);
      temp_58_0002.freeRef();
      return temp_58_0001;
    }

  }

  /**
   * The type Ll.
   */
  public static class LL extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer() {
      ImgTileSelectLayer temp_58_0004 = new ImgTileSelectLayer(4, 3, 4, 0);
      temp_58_0004.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0003 = RefUtil.addRef(temp_58_0004);
      temp_58_0004.freeRef();
      return temp_58_0003;
    }
  }

  /**
   * The type Ur.
   */
  public static class UR extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer() {
      ImgTileSelectLayer temp_58_0006 = new ImgTileSelectLayer(4, 3, 0, 3);
      temp_58_0006.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0005 = RefUtil.addRef(temp_58_0006);
      temp_58_0006.freeRef();
      return temp_58_0005;
    }

  }

  /**
   * The type Lr.
   */
  public static class LR extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer() {
      ImgTileSelectLayer temp_58_0008 = new ImgTileSelectLayer(4, 3, 4, 3);
      temp_58_0008.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0007 = RefUtil.addRef(temp_58_0008);
      temp_58_0008.freeRef();
      return temp_58_0007;
    }

  }

}
