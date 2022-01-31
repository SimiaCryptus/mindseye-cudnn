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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.function.Function;

/**
 * The type Exploded convolution leg.
 */
class ExplodedConvolutionLeg extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionLeg.class);

  /**
   * The Convolution params.
   */
  @Nonnull
  public final ConvolutionParams convolutionParams;
  /**
   * The Sub layers.
   */
  @Nonnull
  public final RefList<Layer> subLayers;
  /**
   * The Sub kernels.
   */
  @Nonnull
  public final RefList<SimpleConvolutionLayer> subKernels = new RefArrayList<>();
  /**
   * The From band.
   */
  public final int fromBand;
  /**
   * The To band.
   */
  public final int toBand;

  /**
   * Instantiates a new Exploded convolution leg.
   *
   * @param convolutionParams the convolution params
   * @param fromBand          the from band
   * @param toBand            the to band
   */
  public ExplodedConvolutionLeg(@Nonnull ConvolutionParams convolutionParams, int fromBand, int toBand) {
    this.fromBand = fromBand;
    this.toBand = toBand;
    this.convolutionParams = convolutionParams;
    this.subLayers = new RefArrayList<>();
    int inputBands = getInputBands();
    final int inputBandsSq = inputBands * inputBands;
    @Nonnull final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
        this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      int paddingX = (convolutionParams.masterFilterDimensions[0] - 1) / 2;
      int paddingY = (convolutionParams.masterFilterDimensions[1] - 1) / 2;
      SimpleConvolutionLayer convolutionLayer = new SimpleConvolutionLayer(filterDimensions[0], filterDimensions[1], inputBandsSq);
      convolutionLayer.setStrideX(this.convolutionParams.strideX);
      convolutionLayer.setStrideY(this.convolutionParams.strideY);
      convolutionLayer.setPrecision(this.convolutionParams.precision);
      PipelineNetwork stackableConv = new PipelineNetwork(1);
      stackableConv.setName(String.format("Bands(%d to %d)", offset / inputBands, (offset + inputBandsSq) / inputBands));
      if (paddingY != 0 || paddingX != 0)
        stackableConv.add(new ImgZeroPaddingLayer(paddingX, paddingY)).freeRef();
      RefUtil.freeRef(stackableConv.add(convolutionLayer.addRef()));
      if (paddingY != 0 || paddingX != 0) {
        final Layer nextHead = new ImgZeroPaddingLayer(-paddingX, -paddingY);
        RefUtil.freeRef(stackableConv.add(nextHead.addRef()));
        nextHead.freeRef();
      }
      Precision precision = convolutionLayer.getPrecision();
      int[] kernelDimensions = convolutionLayer.getKernelDimensions();
      subKernels.add(convolutionLayer);
      this.subLayers.add(getTileSubnet(stackableConv,
          Math.max(filterDimensions[0], filterDimensions[1]), kernelDimensions, precision));
    }
  }

  /**
   * Gets input bands.
   *
   * @return the input bands
   */
  public int getInputBands() {
    return this.toBand - this.fromBand;
  }

  /**
   * Write.
   *
   * @param filter the filter
   */
  public void write(@Nonnull Tensor filter) {
    assert filter.rms() > 0;
    int inputBands = getInputBands();
    @Nonnull final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
        this.convolutionParams.masterFilterDimensions.length);
    int outputBands = this.convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : RefString.format("%d >= %d", squareOutputBands,
        convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : RefString.format("%d %% %d", squareOutputBands, inputBands);
    filterDimensions[2] = inputBands * outputBands;
    assert RefArrays.equals(filter.getDimensions(), filterDimensions) : RefArrays.toString(filter.getDimensions())
        + " != " + RefArrays.toString(filterDimensions);
    final int inputBandsSq = inputBands * inputBands;
    assert subLayers.size() > 0;
    RefIntStream.range(0, subLayers.size()).parallel().forEach(layerNumber -> {
      final int filterBandOffset = layerNumber * inputBandsSq;
      Tensor kernel = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq);
      kernel.setByCoord(c -> {
        int[] coords = c.getCoords();
        int filterBand = getFilterBand(filterBandOffset, coords[2], squareOutputBands);
        if (filterBand < filterDimensions[2]) {
          return filter.get(coords[0], coords[1], filterBand);
        } else {
          return 0;
        }
      }, true);
      assert kernel.rms() > 0;
      SimpleConvolutionLayer simpleConvolutionLayer = subKernels.get(layerNumber);
      simpleConvolutionLayer.set(kernel);
      simpleConvolutionLayer.freeRef();
    });
    filter.freeRef();
  }

  /**
   * Read tensor.
   *
   * @param extractor the extractor
   * @return the tensor
   */
  @Nonnull
  public Tensor read(@Nonnull @RefAware Function<SimpleConvolutionLayer, Tensor> extractor) {
    int inputBands = getInputBands();
    @Nonnull final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
        this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    int outputBands = convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : RefString.format("%d >= %d", squareOutputBands,
        convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : RefString.format("%d %% %d", squareOutputBands, inputBands);
    @Nonnull
    Tensor resultDelta = new Tensor(filterDimensions[0], filterDimensions[1], inputBands * outputBands);

    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      int _layerNumber = layerNumber;
      Tensor deltaTensor = extractor.apply(subKernels.get(layerNumber));
      if (null != deltaTensor) {
        deltaTensor.forEach(RefUtil.wrapInterface((v, c) -> {
          int[] coords = c.getCoords();
          int filterBand = getFilterBand(_layerNumber * inputBands * inputBands, coords[2], squareOutputBands);
          if (filterBand < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], filterBand, v);
          }
        }, resultDelta.addRef()), false);
      }
      if (null != deltaTensor)
        deltaTensor.freeRef();
    }
    RefUtil.freeRef(extractor);
    return resultDelta;
  }

  /**
   * Gets filter band.
   *
   * @param filterBandOffset  the filter band offset
   * @param cellFilterBand    the cell filter band
   * @param squareOutputBands the square output bands
   * @return the filter band
   */
  public int getFilterBand(int filterBandOffset, int cellFilterBand, int squareOutputBands) {
    int inputBands = getInputBands();
    assert cellFilterBand >= 0;
    assert cellFilterBand < inputBands * inputBands;
    assert filterBandOffset < inputBands * squareOutputBands;
    int filterBand = cellFilterBand + filterBandOffset;
    filterBand = Coordinate.transposeXY(inputBands, convolutionParams.outputBands, filterBand);
    return filterBand;
  }

  /**
   * Read tensor.
   *
   * @param deltaSet the delta set
   * @param remove   the remove
   * @return the tensor
   */
  @Nonnull
  public Tensor read(@Nonnull DeltaSet<UUID> deltaSet, boolean remove) {
    return read(RefUtil.wrapInterface(sublayer -> {
      RefMap<UUID, Delta<UUID>> map = deltaSet.getMap();
      Delta<UUID> uuidDelta = map.get(sublayer.getId());
      assert uuidDelta != null;
      final Delta<UUID> subnetDelta;
      if (remove) {
        subnetDelta = map.remove(sublayer.addRef());
        uuidDelta.freeRef();
      } else {
        subnetDelta = uuidDelta;
      }
      map.freeRef();
      if (null == subnetDelta) {
        String toString = sublayer.toString();
        sublayer.freeRef();
        throw new RuntimeException("No Delta for " + toString);
      }
      Tensor kernel = new Tensor(subnetDelta.getDelta(), sublayer.getKernelDimensions());
      sublayer.freeRef();
      subnetDelta.freeRef();
      return kernel;
    }, deltaSet));
  }

  /**
   * Read tensor.
   *
   * @return the tensor
   */
  @Nonnull
  public Tensor read() {
    return read(sublayer -> {
      Tensor kernel = sublayer.getKernel();
      sublayer.freeRef();
      return kernel;
    });
  }

  /**
   * Add dag node.
   *
   * @param input   the input
   * @param network the network
   * @return the dag node
   */
  @Nullable
  public DAGNode add(@Nonnull final DAGNode input, DAGNetwork network) {
    assertAlive();
    if (getInputBands() == this.convolutionParams.outputBands) {
      assert 1 == subLayers.size();
      assert network != null;
      InnerNode node = network.add(subLayers.get(0), input);
      network.freeRef();
      return node;
    } else {
      ImgConcatLayer concatLayer = new ImgConcatLayer();
      concatLayer.setMaxBands(this.convolutionParams.outputBands);
      concatLayer.setPrecision(this.convolutionParams.precision);
      assert network != null;
      concatLayer.setParallel(CudaSettings.INSTANCE().conv_para_2);
      InnerNode node = network.add(concatLayer,
          subLayers.stream().map(RefUtil.wrapInterface((Function<? super Layer, ? extends InnerNode>) layer -> {
            return network.add(layer, input.addRef());
          }, network, input)).toArray(i -> new DAGNode[i]));
      node.setParallel(CudaSettings.INSTANCE().conv_para_2);
      return node;
    }
  }

  @Nonnull
  @Override
  public String toString() {
    return "ExplodedConvolutionLeg{" + "fromBand=" + fromBand + ", toBand=" + toBand + '}';
  }

  public void _free() {
    subKernels.freeRef();
    subLayers.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ExplodedConvolutionLeg addRef() {
    return (ExplodedConvolutionLeg) super.addRef();
  }

  @Nonnull
  private ImgTileSubnetLayer getTileSubnet(@Nullable final Layer network, final int bands, final int[] kernelDimensions,
                                           final Precision precision) {
    int maxSize = (int) Math.sqrt(CudaSettings.INSTANCE().maxIoElements / bands);
    int width = kernelDimensions[0];
    int height = kernelDimensions[1];
    int strideX = maxSize - (width - 1) / 2;
    int strideY = maxSize - (height - 1) / 2;
    ImgTileSubnetLayer subnetLayer = new ImgTileSubnetLayer(network, maxSize, maxSize, strideX, strideY);
    subnetLayer.setParallel(CudaSettings.INSTANCE().conv_para_3);
    subnetLayer.setPrecision(precision);
    return subnetLayer;
  }
}
