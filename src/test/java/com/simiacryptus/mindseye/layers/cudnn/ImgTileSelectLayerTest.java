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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class ImgTileSelectLayerTest extends CudnnLayerTestBase {

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

  public static @SuppressWarnings("unused")
  ImgTileSelectLayerTest[] addRefs(ImgTileSelectLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayerTest::addRef)
        .toArray((x) -> new ImgTileSelectLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileSelectLayerTest[][] addRefs(ImgTileSelectLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayerTest::addRefs)
        .toArray((x) -> new ImgTileSelectLayerTest[x][]);
  }

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

  public @Override
  @SuppressWarnings("unused")
  ImgTileSelectLayerTest addRef() {
    return (ImgTileSelectLayerTest) super.addRef();
  }

  public static @RefAware
  class UL extends ImgTileSelectLayerTest {

    public static @SuppressWarnings("unused")
    UL[] addRefs(UL[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(UL::addRef).toArray((x) -> new UL[x]);
    }

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0002 = new ImgTileSelectLayer(4, 3, 0, 0);
      ImgTileSelectLayer temp_58_0001 = temp_58_0002
          .setPrecision(Precision.Double);
      if (null != temp_58_0002)
        temp_58_0002.freeRef();
      return temp_58_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    UL addRef() {
      return (UL) super.addRef();
    }

  }

  public static @RefAware
  class LL extends ImgTileSelectLayerTest {

    public static @SuppressWarnings("unused")
    LL[] addRefs(LL[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(LL::addRef).toArray((x) -> new LL[x]);
    }

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0004 = new ImgTileSelectLayer(4, 3, 4, 0);
      ImgTileSelectLayer temp_58_0003 = temp_58_0004
          .setPrecision(Precision.Double);
      if (null != temp_58_0004)
        temp_58_0004.freeRef();
      return temp_58_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    LL addRef() {
      return (LL) super.addRef();
    }

  }

  public static @RefAware
  class UR extends ImgTileSelectLayerTest {

    public static @SuppressWarnings("unused")
    UR[] addRefs(UR[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(UR::addRef).toArray((x) -> new UR[x]);
    }

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0006 = new ImgTileSelectLayer(4, 3, 0, 3);
      ImgTileSelectLayer temp_58_0005 = temp_58_0006
          .setPrecision(Precision.Double);
      if (null != temp_58_0006)
        temp_58_0006.freeRef();
      return temp_58_0005;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    UR addRef() {
      return (UR) super.addRef();
    }

  }

  public static @RefAware
  class LR extends ImgTileSelectLayerTest {

    public static @SuppressWarnings("unused")
    LR[] addRefs(LR[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(LR::addRef).toArray((x) -> new LR[x]);
    }

    @Nonnull
    @Override
    public ImgTileSelectLayer getLayer(final int[][] inputSize, Random random) {
      ImgTileSelectLayer temp_58_0008 = new ImgTileSelectLayer(4, 3, 4, 3);
      ImgTileSelectLayer temp_58_0007 = temp_58_0008
          .setPrecision(Precision.Double);
      if (null != temp_58_0008)
        temp_58_0008.freeRef();
      return temp_58_0007;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    LR addRef() {
      return (LR) super.addRef();
    }

  }

}
