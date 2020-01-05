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
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class ImgLinearSubnetLayerTest extends CudnnLayerTestBase {

  private final Layer layer1 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final Layer layer2 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final Layer layer3 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final int smallSize;
  private final int largeSize;

  public ImgLinearSubnetLayerTest() {
    testingBatchSize = 10;
    smallSize = 2;
    largeSize = 100;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  public static @SuppressWarnings("unused")
  ImgLinearSubnetLayerTest[] addRefs(ImgLinearSubnetLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgLinearSubnetLayerTest::addRef)
        .toArray((x) -> new ImgLinearSubnetLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgLinearSubnetLayerTest[][] addRefs(ImgLinearSubnetLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgLinearSubnetLayerTest::addRefs)
        .toArray((x) -> new ImgLinearSubnetLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, 3}};
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{largeSize, largeSize, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgLinearSubnetLayer().add(0, 1, layer1).add(1, 2, layer2).add(2, 3, layer3);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgLinearSubnetLayerTest addRef() {
    return (ImgLinearSubnetLayerTest) super.addRef();
  }

  public static @RefAware
  class Basic extends ImgLinearSubnetLayerTest {

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }

  }

}
