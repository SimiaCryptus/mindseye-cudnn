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

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgTileSubnetLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.IntStream;

class ExplodedConvolutionLeg extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionLeg.class);

  public final ConvolutionParams convolutionParams;
  @Nonnull
  public final List<Layer> subLayers;
  @Nonnull
  public final List<SimpleConvolutionLayer> subKernels = new ArrayList<>();
  public final int fromBand;
  public final int toBand;

  public ExplodedConvolutionLeg(ConvolutionParams convolutionParams, int fromBand, int toBand) {
    this.fromBand = fromBand;
    this.toBand = toBand;
    this.convolutionParams = convolutionParams;
    this.subLayers = new ArrayList<>();
    int inputBands = getInputBands();
    final int inputBandsSq = inputBands * inputBands;
    @Nonnull final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      int paddingX = (convolutionParams.masterFilterDimensions[0] - 1) / 2;
      int paddingY = (convolutionParams.masterFilterDimensions[1] - 1) / 2;

      SimpleConvolutionLayer simpleConvolutionLayer = new SimpleConvolutionLayer(filterDimensions[0], filterDimensions[1], inputBandsSq) //
          .setStrideX(this.convolutionParams.strideX) //
          .setStrideY(this.convolutionParams.strideY) //
          .setPrecision(this.convolutionParams.precision);

      PipelineNetwork stackableConv = new PipelineNetwork(1);
      if (paddingY != 0 || paddingX != 0) stackableConv.wrap(new ImgZeroPaddingLayer(paddingX, paddingY)).freeRef();
      stackableConv.add(simpleConvolutionLayer).freeRef();
      if (paddingY != 0 || paddingX != 0) stackableConv.wrap(new ImgZeroPaddingLayer(-paddingX, -paddingY)).freeRef();
      subKernels.add(simpleConvolutionLayer);
      this.subLayers.add(getTileSubnet(stackableConv, Math.max(filterDimensions[0], filterDimensions[1]), simpleConvolutionLayer.getKernelDimensions(), simpleConvolutionLayer.getPrecision()));
      stackableConv.freeRef();
      //this.subLayers.add(simpleConvolutionLayer);
    }
  }

  @Nonnull
  private ImgTileSubnetLayer getTileSubnet(final Layer network, final int bands, final int[] kernelDimensions, final Precision precision) {
    int maxSize = (int) Math.sqrt(CudaSettings.INSTANCE().getMaxIoElements() / bands);
    int width = kernelDimensions[0];
    int height = kernelDimensions[1];
    return new ImgTileSubnetLayer(network, maxSize, maxSize, maxSize - ((width - 1) / 2), maxSize - ((height - 1) / 2)).setParallel(CudaSettings.INSTANCE().isConv_para_3()).setPrecision(precision);
  }

  @Override
  protected void _free() {
    this.subLayers.forEach(x -> x.freeRef());
    this.subKernels.forEach(x -> x.freeRef());
    super._free();
  }

  @Nonnull
  public ExplodedConvolutionLeg write(@Nonnull Tensor filter) {
    int inputBands = getInputBands();
    @Nonnull final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    int outputBands = this.convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : String.format("%d >= %d", squareOutputBands, convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : String.format("%d %% %d", squareOutputBands, inputBands);
    filterDimensions[2] = inputBands * outputBands;
    assert Arrays.equals(filter.getDimensions(), filterDimensions) : Arrays.toString(filter.getDimensions()) + " != " + Arrays.toString(filterDimensions);
    final int inputBandsSq = inputBands * inputBands;
    IntStream.range(0, subLayers.size()).parallel().forEach(layerNumber -> {
      final int filterBandOffset = layerNumber * inputBandsSq;
      @Nonnull Tensor kernel = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq).setByCoord(c -> {
        int[] coords = c.getCoords();
        int filterBand = getFilterBand(filterBandOffset, coords[2], squareOutputBands);
        if (filterBand < filterDimensions[2]) {
          return filter.get(coords[0], coords[1], filterBand);
        } else {
          return 0;
        }
      }, true);
      subKernels.get(layerNumber).set(kernel);
      kernel.freeRef();
    });
    return this;
  }

  @Nonnull
  public Tensor read(@Nonnull Function<SimpleConvolutionLayer, Tensor> extractor) {
    int inputBands = getInputBands();
    @Nonnull final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    int outputBands = convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : String.format("%d >= %d", squareOutputBands, convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : String.format("%d %% %d", squareOutputBands, inputBands);
    @Nonnull Tensor resultDelta = new Tensor(filterDimensions[0], filterDimensions[1], inputBands * outputBands);

    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      int _layerNumber = layerNumber;
      Tensor deltaTensor = extractor.apply((subKernels.get(layerNumber)));
      if (null != deltaTensor) {
        deltaTensor.forEach((v, c) -> {
          int[] coords = c.getCoords();
          int filterBand = getFilterBand(_layerNumber * inputBands * inputBands, coords[2], squareOutputBands);
          if (filterBand < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], filterBand, v);
          }
        }, false);
        deltaTensor.freeRef();
      }
    }
    return resultDelta;
  }

  public int getFilterBand(int filterBandOffset, int cellFilterBand, int squareOutputBands) {
    int inputBands = getInputBands();
    assert cellFilterBand >= 0;
    assert cellFilterBand < (inputBands * inputBands);
    assert filterBandOffset < (inputBands * squareOutputBands);
    int filterBand = cellFilterBand + filterBandOffset;
    filterBand = Coordinate.transposeXY(inputBands, convolutionParams.outputBands, filterBand);
    return filterBand;
  }

  public int getInputBands() {
    return this.toBand - this.fromBand;
  }

  @Nonnull
  public Tensor read(@Nonnull DeltaSet<UUID> deltaSet, boolean remove) {
    return read((sublayer) -> {
      final Delta<UUID> subnetDelta = remove ? deltaSet.getMap().remove(sublayer) : deltaSet.getMap().get(sublayer.getId());
      if (null == subnetDelta) throw new RuntimeException("No Delta for " + sublayer);
      double[] delta = subnetDelta.getDelta();
      return new Tensor(delta, sublayer.kernel.getDimensions());
    });
  }

  @Nonnull
  public Tensor read() {
    return read((sublayer) -> {
      Tensor kernel = sublayer.kernel;
      kernel.addRef();
      return kernel;
    });
  }

  public DAGNode add(@Nonnull final DAGNode input) {
    assertAlive();
    DAGNetwork network = input.getNetwork();
    final int[] filterDimensions = this.convolutionParams.masterFilterDimensions;
    if (getInputBands() == this.convolutionParams.outputBands) {
      assert 1 == subLayers.size();
      return network.add(subLayers.get(0), input);
    } else {
      InnerNode newHead = network.wrap(new ImgConcatLayer()
              .setMaxBands(this.convolutionParams.outputBands)
              .setPrecision(this.convolutionParams.precision)
              .setParallel(CudaSettings.INSTANCE().isConv_para_2()),
          subLayers.stream().map(l -> network.add(l, input.addRef())).toArray(i -> new DAGNode[i])).setParallel(CudaSettings.INSTANCE().isConv_para_2());
      input.freeRef();
      return newHead;
    }
  }

  @Nonnull
  @Override
  public String toString() {
    return "ExplodedConvolutionLeg{" +
        "fromBand=" + fromBand +
        ", toBand=" + toBand +
        '}';
  }
}
