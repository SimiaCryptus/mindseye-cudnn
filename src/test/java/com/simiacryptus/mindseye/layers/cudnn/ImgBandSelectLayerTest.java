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
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class ImgBandSelectLayerTest extends CudnnLayerTestBase {

  final Precision precision;
  final ImgBandSelectLayer layer;
  final int inputBands;
  private final int smallSize;
  private final int largeSize;

  public ImgBandSelectLayerTest(final Precision precision, int inputBands, final int fromBand, int toBand) {
    this.precision = precision;
    {
      ImgBandSelectLayer temp_20_0002 = new ImgBandSelectLayer(fromBand, toBand);
      ImgBandSelectLayer temp_20_0001 = temp_20_0002.setPrecision(precision);
      if (null != temp_20_0002)
        temp_20_0002.freeRef();
      layer = temp_20_0001 == null ? null : temp_20_0001.addRef();
      if (null != temp_20_0001)
        temp_20_0001.freeRef();
    }
    this.inputBands = inputBands;
    smallSize = 2;
    largeSize = 1000;
    testingBatchSize = 1;
  }

  @Override
  public Layer getReferenceLayer() {
    return layer.getCompatibilityLayer();
  }

  public static @SuppressWarnings("unused")
  ImgBandSelectLayerTest[] addRefs(ImgBandSelectLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandSelectLayerTest::addRef)
        .toArray((x) -> new ImgBandSelectLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgBandSelectLayerTest[][] addRefs(ImgBandSelectLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandSelectLayerTest::addRefs)
        .toArray((x) -> new ImgBandSelectLayerTest[x][]);
  }

  @Override
  public void run(@NotNull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    //    CudaSystem.addLog(apiLog);
    super.run(log);
    //    apiLog.close();
    //    CudaSystem.apiLog.remove(apiLog);
  }

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

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, inputBands}};
  }

  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{largeSize, largeSize, inputBands}};
  }

  public @SuppressWarnings("unused")
  void _free() {
    if (null != layer)
      layer.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  ImgBandSelectLayerTest addRef() {
    return (ImgBandSelectLayerTest) super.addRef();
  }

  public static @RefAware
  class Double extends ImgBandSelectLayerTest {
    public Double() {
      super(Precision.Double, 5, 2, 4);
    }

    public static @SuppressWarnings("unused")
    Double[] addRefs(Double[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Double::addRef).toArray((x) -> new Double[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Double addRef() {
      return (Double) super.addRef();
    }
  }

  public static @RefAware
  class Float extends ImgBandSelectLayerTest {
    public Float() {
      super(Precision.Float, 2, 0, 1);
    }

    public static @SuppressWarnings("unused")
    Float[] addRefs(Float[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Float::addRef).toArray((x) -> new Float[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Float addRef() {
      return (Float) super.addRef();
    }
  }

}
