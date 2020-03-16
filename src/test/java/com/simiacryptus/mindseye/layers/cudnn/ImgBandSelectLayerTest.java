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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import org.junit.jupiter.api.AfterEach;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public abstract class ImgBandSelectLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  @Nonnull
  @RefIgnore
  final ImgBandSelectLayer layer;
  final int inputBands;
  private final int smallSize;
  private final int largeSize;

  public ImgBandSelectLayerTest(final Precision precision, int inputBands, final int fromBand, int toBand) {
    this.precision = precision;
    ImgBandSelectLayer temp_20_0002 = new ImgBandSelectLayer(fromBand, toBand);
    temp_20_0002.setPrecision(precision);
    ImgBandSelectLayer temp_20_0001 = RefUtil.addRef(temp_20_0002);
    temp_20_0002.freeRef();
    layer = temp_20_0001.addRef();
    temp_20_0001.freeRef();
    this.inputBands = inputBands;
    smallSize = 2;
    largeSize = 1000;
    testingBatchSize = 1;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{largeSize, largeSize, inputBands}};
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

  //  /**
  //   * Basic 64-bit apply
  //   */
  //  public static class BigDouble extends ImgBandSelectLayerTest {
  //    /**
  //     * Instantiates a new Double.
  //     */
  //    public BigDouble() {
  //      super(Precision.Double, 1024, 0, 256);
  //    }
  //  }

  @Nullable
  @Override
  public Layer getLayer() {
    return layer.addRef();
  }

  @Override
  public Layer getReferenceLayer() {
    return layer.getCompatibilityLayer();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{smallSize, smallSize, inputBands}};
  }

  @AfterEach
  void cleanup() {
    if (null != layer)
      layer.freeRef();
  }

  public static class Double extends ImgBandSelectLayerTest {
    public Double() {
      super(Precision.Double, 5, 2, 4);
    }
  }

  public static class Float extends ImgBandSelectLayerTest {
    public Float() {
      super(Precision.Float, 2, 0, 1);
    }
  }

}
