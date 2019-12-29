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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.notebook.NotebookOutput;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

public abstract class GramianLayerTest extends CudnnLayerTestBase {

  public GramianLayerTest() {
    this.tolerance = 1e-2;
    testingBatchSize = 1;
  }

  @Override
  public Layer getReferenceLayer() {
    return new LayerBase() {

      @Nullable
      @Override
      public Result eval(Result... array) {
        Tensor input = array[0].getData().get(0);
        int[] inputDimensions = input.getDimensions();
        int inBands = inputDimensions[2];
        Tensor output = new Tensor(1, 1, inBands * inBands);
        output.setByCoord(c -> {
          int[] coords = c.getCoords();
          int outBand = coords[2];
          int bandA = outBand / inBands;
          int bandB = outBand % inBands;
          return IntStream.range(0, inputDimensions[0]).mapToDouble(x -> {
            return IntStream.range(0, inputDimensions[1]).mapToDouble(y -> {
              return input.get(x, y, bandA) * input.get(x, y, bandB);
            }).average().getAsDouble();
          }).average().getAsDouble();
        });
        return new Result(new TensorArray(output), (a, b) -> {
        });
      }

      @Override
      public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
        return null;
      }

      @Nullable
      @Override
      public List<double[]> state() {
        return null;
      }
    };
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return GramianLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {2, 2, 3}
    };
  }

  @Override
  public abstract int[][] getLargeDims(final Random random);

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new GramianLayer();
  }

  @Override
  public void run(@NotNull NotebookOutput log) {
//    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }

  public static class Image extends GramianLayerTest {
    public Image() {
      super();
    }

    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{
          {1000, 1000, 3}
      };
    }

  }

  public static class Deep extends GramianLayerTest {
    public Deep() {
      super();
    }

    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{
          {100, 100, 512}
      };
    }
  }


}
