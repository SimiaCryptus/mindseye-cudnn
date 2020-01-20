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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class ImgTileSelectLayerTest extends CudnnLayerTestBase {

  public ImgTileSelectLayerTest() {
    validateBatchExecution = false;
  }

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
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 6, 1}};
  }

  @Nonnull
  @Override
  public abstract ImgTileSelectLayer getLayer(final int[][] inputSize, Random random);

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileSelectLayerTest addRef() {
    return (ImgTileSelectLayerTest) super.addRef();
  }

  public static class UL extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0002 = new ImgTileSelectLayer(4, 3, 0, 0);
      temp_58_0002.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0001 = RefUtil.addRef(temp_58_0002);
      temp_58_0002.freeRef();
      return temp_58_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    UL addRef() {
      return (UL) super.addRef();
    }
  }

  public static class LL extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0004 = new ImgTileSelectLayer(4, 3, 4, 0);
      temp_58_0004.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0003 = RefUtil.addRef(temp_58_0004);
      temp_58_0004.freeRef();
      return temp_58_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    LL addRef() {
      return (LL) super.addRef();
    }
  }

  public static class UR extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0006 = new ImgTileSelectLayer(4, 3, 0, 3);
      temp_58_0006.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0005 = RefUtil.addRef(temp_58_0006);
      temp_58_0006.freeRef();
      return temp_58_0005;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    UR addRef() {
      return (UR) super.addRef();
    }
  }

  public static class LR extends ImgTileSelectLayerTest {

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0008 = new ImgTileSelectLayer(4, 3, 4, 3);
      temp_58_0008.setPrecision(Precision.Double);
      ImgTileSelectLayer temp_58_0007 = RefUtil.addRef(temp_58_0008);
      temp_58_0008.freeRef();
      return temp_58_0007;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    LR addRef() {
      return (LR) super.addRef();
    }
  }

}
