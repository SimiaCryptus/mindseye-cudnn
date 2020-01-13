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
import java.util.Arrays;
import java.util.UUID;
import java.util.function.Function;

class ExplodedConvolutionLeg extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionLeg.class);

  public final ConvolutionParams convolutionParams;
  @Nonnull
  public final RefList<Layer> subLayers;
  @Nonnull
  public final RefList<SimpleConvolutionLayer> subKernels = new RefArrayList<>();
  public final int fromBand;
  public final int toBand;

  public ExplodedConvolutionLeg(ConvolutionParams convolutionParams, int fromBand, int toBand) {
    this.fromBand = fromBand;
    this.toBand = toBand;
    this.convolutionParams = convolutionParams;
    RefList<Layer> temp_02_0001 = new RefArrayList<>();
    this.subLayers = temp_02_0001 == null ? null : temp_02_0001.addRef();
    if (null != temp_02_0001)
      temp_02_0001.freeRef();
    int inputBands = getInputBands();
    final int inputBandsSq = inputBands * inputBands;
    @Nonnull
    final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
        this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      int paddingX = (convolutionParams.masterFilterDimensions[0] - 1) / 2;
      int paddingY = (convolutionParams.masterFilterDimensions[1] - 1) / 2;

      SimpleConvolutionLayer temp_02_0010 = new SimpleConvolutionLayer(filterDimensions[0], filterDimensions[1],
          inputBandsSq) //
      ;
      SimpleConvolutionLayer temp_02_0014 = temp_02_0010.setStrideX(this.convolutionParams.strideX) //
      ;
      SimpleConvolutionLayer temp_02_0015 = temp_02_0014.setStrideY(this.convolutionParams.strideY) //
      ;
      SimpleConvolutionLayer simpleConvolutionLayer = temp_02_0015.setPrecision(this.convolutionParams.precision);

      if (null != temp_02_0015)
        temp_02_0015.freeRef();
      if (null != temp_02_0014)
        temp_02_0014.freeRef();
      if (null != temp_02_0010)
        temp_02_0010.freeRef();
      PipelineNetwork stackableConv = new PipelineNetwork(1);
      if (paddingY != 0 || paddingX != 0)
        stackableConv.add(new ImgZeroPaddingLayer(paddingX, paddingY));
      RefUtil.freeRef(stackableConv.add(simpleConvolutionLayer == null ? null : simpleConvolutionLayer.addRef()));
      if (paddingY != 0 || paddingX != 0) {
        final Layer nextHead = new ImgZeroPaddingLayer(-paddingX, -paddingY);
        RefUtil.freeRef(stackableConv.add(nextHead == null ? null : nextHead.addRef()));
        if (null != nextHead)
          nextHead.freeRef();
      }
      subKernels.add(simpleConvolutionLayer == null ? null : simpleConvolutionLayer.addRef());
      this.subLayers.add(getTileSubnet(stackableConv == null ? null : stackableConv.addRef(),
          Math.max(filterDimensions[0], filterDimensions[1]), simpleConvolutionLayer.getKernelDimensions(),
          simpleConvolutionLayer.getPrecision()));
      if (null != stackableConv)
        stackableConv.freeRef();
      if (null != simpleConvolutionLayer)
        simpleConvolutionLayer.freeRef();
    }
  }

  public int getInputBands() {
    return this.toBand - this.fromBand;
  }

  public static @SuppressWarnings("unused") ExplodedConvolutionLeg[] addRefs(ExplodedConvolutionLeg[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ExplodedConvolutionLeg::addRef)
        .toArray((x) -> new ExplodedConvolutionLeg[x]);
  }

  public static @SuppressWarnings("unused") ExplodedConvolutionLeg[][] addRefs(ExplodedConvolutionLeg[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ExplodedConvolutionLeg::addRefs)
        .toArray((x) -> new ExplodedConvolutionLeg[x][]);
  }

  @Nonnull
  public void write(@Nonnull Tensor filter) {
    int inputBands = getInputBands();
    @Nonnull
    final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
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
    RefIntStream.range(0, subLayers.size()).parallel().forEach(RefUtil.wrapInterface(layerNumber -> {
      final int filterBandOffset = layerNumber * inputBandsSq;
      Tensor temp_02_0011 = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq);
      @Nonnull
      Tensor kernel = temp_02_0011.setByCoord(RefUtil.wrapInterface(c -> {
        int[] coords = c.getCoords();
        int filterBand = getFilterBand(filterBandOffset, coords[2], squareOutputBands);
        if (filterBand < filterDimensions[2]) {
          return filter.get(coords[0], coords[1], filterBand);
        } else {
          return 0;
        }
      }, filter == null ? null : filter.addRef()), true);
      if (null != temp_02_0011)
        temp_02_0011.freeRef();
      SimpleConvolutionLayer temp_02_0016 = subKernels.get(layerNumber);
      temp_02_0016.set(kernel == null ? null : kernel);
      if (null != temp_02_0016)
        temp_02_0016.freeRef();
    }, filter == null ? null : filter));
  }

  @Nonnull
  public Tensor read(@Nonnull Function<SimpleConvolutionLayer, Tensor> extractor) {
    int inputBands = getInputBands();
    @Nonnull
    final int[] filterDimensions = RefArrays.copyOf(this.convolutionParams.masterFilterDimensions,
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
      Tensor deltaTensor = extractor.apply((subKernels.get(layerNumber)));
      if (null != deltaTensor) {
        deltaTensor.forEach(RefUtil.wrapInterface((v, c) -> {
          int[] coords = c.getCoords();
          int filterBand = getFilterBand(_layerNumber * inputBands * inputBands, coords[2], squareOutputBands);
          if (filterBand < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], filterBand, v);
          }
        }, resultDelta == null ? null : resultDelta.addRef()), false);
      }
      if (null != deltaTensor)
        deltaTensor.freeRef();
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

  @Nonnull
  public Tensor read(@Nonnull DeltaSet<UUID> deltaSet, boolean remove) {
    Tensor temp_02_0008 = read(RefUtil.wrapInterface((sublayer) -> {
      RefMap<UUID, Delta<UUID>> temp_02_0017 = deltaSet.getMap();
      RefMap<UUID, Delta<UUID>> temp_02_0018 = deltaSet.getMap();
      Delta<UUID> temp_02_0019 = temp_02_0018.get(sublayer.getId());
      final Delta<UUID> subnetDelta = remove ? temp_02_0017.remove(sublayer == null ? null : sublayer.addRef())
          : temp_02_0019.addRef();
      if (null != temp_02_0019)
        temp_02_0019.freeRef();
      if (null != temp_02_0018)
        temp_02_0018.freeRef();
      if (null != temp_02_0017)
        temp_02_0017.freeRef();
      if (null == subnetDelta) {
        RuntimeException temp_02_0003 = new RuntimeException("No Delta for " + sublayer);
        if (null != sublayer)
          sublayer.freeRef();
        if (null != subnetDelta)
          subnetDelta.freeRef();
        throw temp_02_0003;
      }
      double[] delta = subnetDelta.getDelta();
      if (null != subnetDelta)
        subnetDelta.freeRef();
      Tensor temp_02_0002 = new Tensor(delta, sublayer.kernel.getDimensions());
      if (null != sublayer)
        sublayer.freeRef();
      return temp_02_0002;
    }, deltaSet == null ? null : deltaSet));
    return temp_02_0008;
  }

  @Nonnull
  public Tensor read() {
    return read((sublayer) -> {
      Tensor temp_02_0004 = sublayer.kernel;
      if (null != sublayer)
        sublayer.freeRef();
      return temp_02_0004;
    });
  }

  public DAGNode add(@Nonnull final DAGNode input) {
    assertAlive();
    DAGNetwork network = input.getNetwork();
    final int[] filterDimensions = this.convolutionParams.masterFilterDimensions;
    if (getInputBands() == this.convolutionParams.outputBands) {
      assert 1 == subLayers.size();
      InnerNode temp_02_0005 = network.add(subLayers.get(0), input == null ? null : input);
      if (null != network)
        network.freeRef();
      return temp_02_0005;
    } else {
      ImgConcatLayer temp_02_0012 = new ImgConcatLayer();
      ImgConcatLayer temp_02_0020 = temp_02_0012.setMaxBands(this.convolutionParams.outputBands);
      ImgConcatLayer temp_02_0021 = temp_02_0020.setPrecision(this.convolutionParams.precision);
      InnerNode temp_02_0022 = network.add(temp_02_0021.setParallel(CudaSettings.INSTANCE().isConv_para_2()),
          subLayers.stream().map(RefUtil.wrapInterface((Function<? super Layer, ? extends InnerNode>) l -> {
            InnerNode temp_02_0007 = network.add(l == null ? null : l.addRef(), input == null ? null : input.addRef());
            if (null != l)
              l.freeRef();
            return temp_02_0007;
          }, network == null ? null : network.addRef(), input == null ? null : input)).toArray(i -> new DAGNode[i]));
      InnerNode temp_02_0006 = temp_02_0022.setParallel(CudaSettings.INSTANCE().isConv_para_2());
      if (null != temp_02_0022)
        temp_02_0022.freeRef();
      if (null != temp_02_0021)
        temp_02_0021.freeRef();
      if (null != temp_02_0020)
        temp_02_0020.freeRef();
      if (null != temp_02_0012)
        temp_02_0012.freeRef();
      if (null != network)
        network.freeRef();
      return temp_02_0006;
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

  public @Override @SuppressWarnings("unused") ExplodedConvolutionLeg addRef() {
    return (ExplodedConvolutionLeg) super.addRef();
  }

  @Nonnull
  private ImgTileSubnetLayer getTileSubnet(final Layer network, final int bands, final int[] kernelDimensions,
      final Precision precision) {
    int maxSize = (int) Math.sqrt(CudaSettings.INSTANCE().getMaxIoElements() / bands);
    int width = kernelDimensions[0];
    int height = kernelDimensions[1];
    ImgTileSubnetLayer temp_02_0013 = new ImgTileSubnetLayer(network == null ? null : network.addRef(), maxSize,
        maxSize, maxSize - ((width - 1) / 2), maxSize - ((height - 1) / 2));
    ImgTileSubnetLayer temp_02_0023 = temp_02_0013.setParallel(CudaSettings.INSTANCE().isConv_para_3());
    ImgTileSubnetLayer temp_02_0009 = temp_02_0023.setPrecision(precision);
    if (null != temp_02_0023)
      temp_02_0023.freeRef();
    if (null != temp_02_0013)
      temp_02_0013.freeRef();
    if (null != network)
      network.freeRef();
    return temp_02_0009;
  }
}
