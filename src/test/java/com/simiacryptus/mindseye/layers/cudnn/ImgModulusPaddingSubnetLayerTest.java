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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract class ImgModulusPaddingSubnetLayerTest extends CudnnLayerTestBase {

  final int modulus;
  final int offset;

  public ImgModulusPaddingSubnetLayerTest(int inputSize, int modulus, int offset) {
    validateBatchExecution = false;
    this.modulus = modulus;
    this.offset = offset;
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  public static @SuppressWarnings("unused") ImgModulusPaddingSubnetLayerTest[] addRefs(
      ImgModulusPaddingSubnetLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgModulusPaddingSubnetLayerTest::addRef)
        .toArray((x) -> new ImgModulusPaddingSubnetLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ImgModulusPaddingSubnetLayerTest[][] addRefs(
      ImgModulusPaddingSubnetLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgModulusPaddingSubnetLayerTest::addRefs)
        .toArray((x) -> new ImgModulusPaddingSubnetLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 2, 2, 1 } };
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 1200, 1200, 1 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgModulusPaddingSubnetLayer(modulus, modulus, offset, offset,
        new ActivationLayer(ActivationLayer.Mode.RELU));
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgModulusPaddingSubnetLayerTest addRef() {
    return (ImgModulusPaddingSubnetLayerTest) super.addRef();
  }

  public static class Basic extends ImgModulusPaddingSubnetLayerTest {
    public Basic() {
      super(2, 3, 0);
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
