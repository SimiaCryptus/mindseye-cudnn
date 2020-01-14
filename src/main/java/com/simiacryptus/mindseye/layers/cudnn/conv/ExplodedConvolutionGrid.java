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

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.layers.cudnn.ImgLinearSubnetLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;

class ExplodedConvolutionGrid extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionGrid.class);

  @Nonnull
  public final RefList<ExplodedConvolutionLeg> subLayers;
  @Nonnull
  public final ConvolutionParams convolutionParams;

  public ExplodedConvolutionGrid(@Nonnull ConvolutionParams convolutionParams, int maxBandBatch) {
    this.convolutionParams = convolutionParams;
    int bandWidth = (maxBandBatch == 0) ? convolutionParams.inputBands : maxBandBatch;
    int rows = (int) Math.ceil((double) convolutionParams.inputBands / bandWidth);
    RefList<ExplodedConvolutionLeg> temp_08_0001 = RefIntStream.range(0, rows).map(x -> x * bandWidth)
        .mapToObj(fromBand -> {
          int toBand = Math.min(convolutionParams.inputBands, fromBand + bandWidth);
          if (fromBand >= toBand)
            throw new RuntimeException(fromBand + " >= " + toBand);
          return new ExplodedConvolutionLeg(convolutionParams, fromBand, toBand);
        }).collect(RefCollectors.toList());
    subLayers = temp_08_0001 == null ? null : temp_08_0001.addRef();
    if (null != temp_08_0001)
      temp_08_0001.freeRef();
  }

  @Nonnull
  public PipelineNetwork getNetwork() {
    assertAlive();
    @Nonnull
    PipelineNetwork network = new PipelineNetwork(1);
    add(network.getInput(0));
    return network;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ExplodedConvolutionGrid[] addRefs(@Nullable ExplodedConvolutionGrid[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ExplodedConvolutionGrid::addRef)
        .toArray((x) -> new ExplodedConvolutionGrid[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ExplodedConvolutionGrid[][] addRefs(@Nullable ExplodedConvolutionGrid[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ExplodedConvolutionGrid::addRefs)
        .toArray((x) -> new ExplodedConvolutionGrid[x][]);
  }

  @Nonnull
  public ExplodedConvolutionGrid write(@Nonnull Tensor filter) {
    if (1 == subLayers.size()) {
      ExplodedConvolutionLeg temp_08_0007 = subLayers.get(0);
      temp_08_0007.write(filter.addRef());
      temp_08_0007.freeRef();
    } else {
      subLayers.forEach(leg -> {
        @Nonnull
        int[] legDims = {convolutionParams.masterFilterDimensions[0], convolutionParams.masterFilterDimensions[1],
            leg.getInputBands() * convolutionParams.outputBands};
        @Nonnull
        Tensor template = new Tensor(legDims);
        @Nullable
        Tensor tensor = template.mapCoords(RefUtil.wrapInterface(c -> {
          int[] coords = c.getCoords();
          return filter.get(coords[0], coords[1], getFilterBand(leg.addRef(), coords[2]));
        }, leg.addRef(), filter.addRef()), false);
        template.freeRef();
        leg.write(tensor.addRef());
        tensor.freeRef();
      });
    }
    filter.freeRef();
    return this.addRef();
  }

  public Tensor read(@Nonnull Function<ExplodedConvolutionLeg, Tensor> extractor) {
    if (1 == subLayers.size()) {
      ExplodedConvolutionLeg temp_08_0009 = subLayers.get(0);
      Tensor temp_08_0008 = extractor.apply(temp_08_0009);
      if (null != temp_08_0009)
        temp_08_0009.freeRef();
      return temp_08_0008;
    } else {
      @Nonnull final Tensor filterDelta = new Tensor(convolutionParams.masterFilterDimensions);
      subLayers.forEach(leg -> {
        Tensor tensor = extractor.apply(leg);
        tensor.forEach(RefUtil.wrapInterface((v, c) -> {
          int[] coords = c.getCoords();
          filterDelta.set(coords[0], coords[1], getFilterBand(leg == null ? null : leg.addRef(), coords[2]), v);
        }, leg == null ? null : leg.addRef(), filterDelta.addRef()), false);
        tensor.freeRef();
      });
      return filterDelta;
    }
  }

  public Tensor read() {
    return read(l -> {
      Tensor temp_08_0002 = l.read();
      l.freeRef();
      return temp_08_0002;
    });
  }

  public Tensor read(@Nonnull DeltaSet<UUID> deltaSet, boolean remove) {
    Tensor temp_08_0004 = read(RefUtil.wrapInterface(l -> {
      Tensor temp_08_0003 = l.read(deltaSet.addRef(), remove);
      l.freeRef();
      return temp_08_0003;
    }, deltaSet));
    return temp_08_0004;
  }

  public void add(@Nonnull DAGNode input) {
    assertAlive();
    DAGNetwork network = input.getNetwork();
    int defaultPaddingX = 0;
    int defaultPaddingY = 0;
    boolean customPaddingX = this.convolutionParams.paddingX != null && convolutionParams.paddingX != defaultPaddingX;
    boolean customPaddingY = this.convolutionParams.paddingY != null && convolutionParams.paddingY != defaultPaddingY;
    final DAGNode paddedInput;
    if (customPaddingX || customPaddingY) {
      int x;
      if (this.convolutionParams.paddingX < -defaultPaddingX) {
        x = this.convolutionParams.paddingX + defaultPaddingX;
      } else if (this.convolutionParams.paddingX > defaultPaddingX) {
        x = this.convolutionParams.paddingX - defaultPaddingX;
      } else {
        x = 0;
      }
      int y;
      if (this.convolutionParams.paddingY < -defaultPaddingY) {
        y = this.convolutionParams.paddingY + defaultPaddingY;
      } else if (this.convolutionParams.paddingY > defaultPaddingY) {
        y = this.convolutionParams.paddingY - defaultPaddingY;
      } else {
        y = 0;
      }
      ImgZeroPaddingLayer temp_08_0005 = new ImgZeroPaddingLayer(x, y);
      assert network != null;
      paddedInput = network.add(temp_08_0005.setPrecision(convolutionParams.precision),
          input.addRef());
      temp_08_0005.freeRef();
    } else {
      paddedInput = input.addRef();
    }
    input.freeRef();
    InnerNode output;
    if (subLayers.size() == 1) {
      ExplodedConvolutionLeg temp_08_0010 = subLayers.get(0);
      output = (InnerNode) temp_08_0010.add(paddedInput.addRef());
      temp_08_0010.freeRef();
    } else {
      ImgLinearSubnetLayer linearSubnetLayer = new ImgLinearSubnetLayer();
      subLayers.forEach(RefUtil.wrapInterface((Consumer<? super ExplodedConvolutionLeg>) leg -> {
        PipelineNetwork subnet = new PipelineNetwork();
        RefUtil.freeRef(leg.add(subnet.getHead()));
        RefUtil.freeRef(linearSubnetLayer.add(leg.fromBand, leg.toBand, subnet.addRef()));
        subnet.freeRef();
        leg.freeRef();
      }, linearSubnetLayer.addRef()));
      boolean isParallel = CudaSettings.INSTANCE().isConv_para_1();
      ImgLinearSubnetLayer temp_08_0011 = linearSubnetLayer.setPrecision(convolutionParams.precision);
      temp_08_0011.setParallel(isParallel);
      temp_08_0011.freeRef();
      assert network != null;
      InnerNode temp_08_0012 = network.add(linearSubnetLayer.addRef(),
          paddedInput.addRef());
      output = temp_08_0012.setParallel(isParallel);
      temp_08_0012.freeRef();
      linearSubnetLayer.freeRef();
    }
    paddedInput.freeRef();
    if (customPaddingX || customPaddingY) {
      int x = !customPaddingX ? 0 : (this.convolutionParams.paddingX - defaultPaddingX);
      int y = !customPaddingY ? 0 : (this.convolutionParams.paddingY - defaultPaddingY);
      if (x > 0)
        x = 0;
      if (y > 0)
        y = 0;
      if (x != 0 || y != 0) {
        ImgZeroPaddingLayer temp_08_0006 = new ImgZeroPaddingLayer(x, y);
        RefUtil.freeRef(network.add(temp_08_0006.setPrecision(convolutionParams.precision),
            output == null ? null : output.addRef()));
        temp_08_0006.freeRef();
      }
    }
    if (null != output)
      output.freeRef();
    if (null != network)
      network.freeRef();
  }

  public void _free() {
    subLayers.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ExplodedConvolutionGrid addRef() {
    return (ExplodedConvolutionGrid) super.addRef();
  }

  private int getFilterBand(@Nonnull ExplodedConvolutionLeg leg, int legFilterBand) {
    int filterBand = legFilterBand;
    filterBand = filterBand + convolutionParams.outputBands * leg.fromBand;
    leg.freeRef();
    return filterBand;
  }

}
