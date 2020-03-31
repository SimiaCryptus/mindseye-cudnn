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
import com.simiacryptus.ref.lang.MustCall;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;

import javax.annotation.Nonnull;

public abstract class RescaledSubnetLayerTest extends CudnnLayerTestBase {

  @Nonnull
  @RefIgnore
  final ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 1, 1);


  public RescaledSubnetLayerTest() {
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{1200, 1200, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    convolutionLayer.set(() -> this.random());
    return new RescaledSubnetLayer(2, convolutionLayer.addRef());
  }

//  @Override
//  public void allTests(@Nonnull NotebookOutput log) {
//    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    //    log.p(log.file((String) null, logName, "GPU Log"));
//    //    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
//    //    CudaSystem.addLog(apiLog);
//    super.allTests(log);
//    //    apiLog.close();
//    //    CudaSystem.apiLog.remove(apiLog);
//  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 1}};
  }

  @AfterEach
  @MustCall
  void cleanup() {
    if (null != convolutionLayer)
      convolutionLayer.freeRef();
  }

  public static class Basic extends RescaledSubnetLayerTest {
  }
}
