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
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public abstract class ImgLinearSubnetLayerTest extends CudnnLayerTestBase {

  @RefIgnore
  private final Layer layer1 = new ActivationLayer(ActivationLayer.Mode.RELU);
  @RefIgnore
  private final Layer layer2 = new ActivationLayer(ActivationLayer.Mode.RELU);
  @RefIgnore
  private final Layer layer3 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final int smallSize;
  private final int largeSize;

  public ImgLinearSubnetLayerTest() {
    testingBatchSize = 10;
    smallSize = 2;
    largeSize = 100;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    ImgLinearSubnetLayer temp_67_0002 = new ImgLinearSubnetLayer();
    final Layer layer5 = layer1 == null ? null : layer1.addRef();
    temp_67_0002.add(0, 1, layer5);
    ImgLinearSubnetLayer temp_67_0003 = temp_67_0002.addRef();
    final Layer layer4 = layer2 == null ? null : layer2.addRef();
    temp_67_0003.add(1, 2, layer4);
    ImgLinearSubnetLayer temp_67_0004 = temp_67_0003.addRef();
    final Layer layer = layer3 == null ? null : layer3.addRef();
    temp_67_0004.add(2, 3, layer);
    ImgLinearSubnetLayer temp_67_0001 = temp_67_0004.addRef();
    temp_67_0004.freeRef();
    temp_67_0003.freeRef();
    temp_67_0002.freeRef();
    return temp_67_0001;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, 3}};
  }

  @AfterEach
  void cleanup() {
    if (null != layer3)
      layer3.freeRef();
    if (null != layer2)
      layer2.freeRef();
    if (null != layer1)
      layer1.freeRef();
  }

  public static class Basic extends ImgLinearSubnetLayerTest {
  }

}
