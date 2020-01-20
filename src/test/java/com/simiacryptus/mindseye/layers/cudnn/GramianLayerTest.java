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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

public abstract class GramianLayerTest extends CudnnLayerTestBase {

  public GramianLayerTest() {
    this.tolerance = 1e-2;
    testingBatchSize = 1;
  }

  @Override
  public Layer getReferenceLayer() {
    return new LayerBase() {

      @Nonnull
      @Override
      public Result eval(@Nullable Result... array) {
        assert array != null;
        TensorList temp_41_0002 = array[0].getData();
        Tensor input = temp_41_0002.get(0);
        temp_41_0002.freeRef();
        ReferenceCounting.freeRefs(array);
        int[] inputDimensions = input.getDimensions();
        int inBands = inputDimensions[2];
        Tensor output = new Tensor(1, 1, inBands * inBands);
        output.setByCoord(RefUtil.wrapInterface(c -> {
              int[] coords = c.getCoords();
              int outBand = coords[2];
              int bandA = outBand / inBands;
              int bandB = outBand % inBands;
              return RefIntStream.range(0, inputDimensions[0]).mapToDouble(RefUtil.wrapInterface(x -> {
                return RefIntStream.range(0, inputDimensions[1]).mapToDouble(RefUtil.wrapInterface(y -> {
                  return input.get(x, y, bandA) * input.get(x, y, bandB);
                }, input.addRef())).average().getAsDouble();
              }, input.addRef())).average().getAsDouble();
            }, input.addRef()));
        input.freeRef();
        Result temp_41_0001 = new Result(new TensorArray(output.addRef()),
            new Result.Accumulator() {
              @Override
              public void accept(@Nullable DeltaSet<UUID> a, @Nullable TensorList b) {
                if (null != b)
                  b.freeRef();
                if (null != a)
                  a.freeRef();
              }

              public @SuppressWarnings("unused")
              void _free() {
              }
            });
        output.freeRef();
        return temp_41_0001;
      }

      @Nullable
      @Override
      public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
        return null;
      }

      @Nullable
      @Override
      public RefList<double[]> state() {
        return null;
      }

      public @SuppressWarnings("unused")
      void _free() {
      }
    };
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return GramianLayer.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  GramianLayerTest[] addRefs(@Nullable GramianLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GramianLayerTest::addRef)
        .toArray((x) -> new GramianLayerTest[x]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Nonnull
  @Override
  public abstract int[][] getLargeDims(final Random random);

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new GramianLayer();
  }

  @Override
  public void run(@Nonnull NotebookOutput log) {
    //    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    //    log.p(log.file((String) null, logName, "GPU Log"));
    //    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GramianLayerTest addRef() {
    return (GramianLayerTest) super.addRef();
  }

  public static class Image extends GramianLayerTest {
    public Image() {
      super();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{1000, 1000, 3}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Image addRef() {
      return (Image) super.addRef();
    }
  }

  public static class Deep extends GramianLayerTest {
    public Deep() {
      super();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{{100, 100, 512}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Deep addRef() {
      return (Deep) super.addRef();
    }
  }

}
