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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgLinearSubnetLayer extends LayerBase
    implements MultiPrecision<ImgLinearSubnetLayer> {

  private static final Logger logger = LoggerFactory.getLogger(ImgLinearSubnetLayer.class);
  private final com.simiacryptus.ref.wrappers.RefList<SubnetLeg> legs = new com.simiacryptus.ref.wrappers.RefArrayList<>();
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
  private boolean parallel = true;

  public ImgLinearSubnetLayer() {
    super();
  }

  protected ImgLinearSubnetLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    setParallel(json.get("parallel").getAsBoolean());
    JsonArray jsonArray = json.get("legs").getAsJsonArray();
    for (int i = 0; i < jsonArray.size(); i++) {
      legs.add(new SubnetLeg(jsonArray.get(i).getAsJsonObject(), rs));
    }
  }

  public com.simiacryptus.ref.wrappers.RefList<SubnetLeg> getLegs() {
    return legs;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ImgLinearSubnetLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }

  public boolean isParallel() {
    return parallel;
  }

  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @SuppressWarnings("unused")
  public static ImgLinearSubnetLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgLinearSubnetLayer(json, rs);
  }

  public ImgLinearSubnetLayer add(final int from, final int to, final Layer layer) {
    getLegs().add(new SubnetLeg(layer, from, to));
    return this;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0];
    TensorList inputData = input.getData();
    @Nonnull
    final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int length = inputData.length();
    int maxBand = legs.stream().mapToInt(x -> x.toBand).max().getAsInt();
    assert maxBand == inputDims[2] : maxBand + " != " + inputDims[2];
    assert com.simiacryptus.ref.wrappers.RefIntStream.range(0, maxBand)
        .allMatch(i -> 1 == legs.stream().filter(x -> x.fromBand <= i && x.toBand > i).count());
    CudaTensor passback = CudaSystem.run(gpu -> {
      return new CudaTensor(gpu.allocate(inputData.getElements() * precision.size, MemoryType.Device, true),
          gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
    });
    AtomicInteger counter = new AtomicInteger(0);
    Result[] legResults;
    legResults = legs.stream().map(leg -> {
      ImgBandSelectLayer imgBandSelectLayer = new ImgBandSelectLayer(leg.fromBand, leg.toBand);
      TensorList legData = imgBandSelectLayer.eval(input).getData();
      //
      //
      //
      //
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      return leg.inner.eval(new Result(legData, (DeltaSet<UUID> ctx, TensorList delta) -> {
        int[] outputDimensions = delta.getDimensions();
        synchronized (passback) {
          CudaSystem.run(gpu -> {
            @Nonnull
            final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision, length,
                outputDimensions[2], outputDimensions[1], outputDimensions[0], //
                inputDims[2] * inputDims[1] * inputDims[0], //
                inputDims[1] * inputDims[0], //
                inputDims[0], //
                1);
            final int byteOffset = viewDescriptor.cStride * leg.fromBand * precision.size;
            assert delta.length() == length;
            assert passback.getDeviceId() == gpu.getDeviceId();
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
            @Nullable
            final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
            @Nonnull
            final CudaMemory passbackBuffer = passback.getMemory(gpu);
            CudaMemory errorPtrMemory = deltaTensor.getMemory(gpu);
            passbackBuffer.synchronize();
            gpu.cudnnTransformTensor(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                errorPtrMemory.getPtr(), precision.getPointer(0.0), viewDescriptor.getPtr(),
                passbackBuffer.getPtr().withByteOffset(byteOffset));
            errorPtrMemory.dirty();
            passbackBuffer.dirty();
          }, passback);
        }
        if (counter.incrementAndGet() >= legs.size()) {
          counter.set(0);
          input.accumulate(ctx, new CudaTensorList(passback, length, inputDims, precision));
        }
      }) {
        public void _free() {
          super._free();
        }
      });
    }).toArray(i -> new Result[i]);
    SumInputsLayer sumInputsLayer = new SumInputsLayer().setParallel(parallel).setPrecision(precision);
    return sumInputsLayer.eval(legResults);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    JsonArray jsonArray = new JsonArray();
    legs.stream().map(x -> x.getJson(resources, dataSerializer)).forEach(jsonArray::add);
    json.add("legs", jsonArray);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    legs.stream().map(x -> x.inner).forEach(x -> x.setFrozen(frozen));
    return super.setFrozen(frozen);
  }

  public void _free() {
    super._free();
  }

  public static @com.simiacryptus.ref.lang.RefAware class SubnetLeg extends ReferenceCountingBase {

    private final Layer inner;
    private final int fromBand;
    private final int toBand;

    public SubnetLeg(final Layer inner, final int fromBand, final int toBand) {
      this.inner = inner;
      this.fromBand = fromBand;
      this.toBand = toBand;
    }

    protected SubnetLeg(@Nonnull final JsonObject json, com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
      fromBand = json.getAsJsonPrimitive("fromBand").getAsInt();
      toBand = json.getAsJsonPrimitive("toBand").getAsInt();
      inner = Layer.fromJson(json.getAsJsonObject("network"), rs);
    }

    @Nonnull
    public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
        DataSerializer dataSerializer) {
      @Nonnull
      final JsonObject json = new JsonObject();
      json.addProperty("fromBand", fromBand);
      json.addProperty("toBand", toBand);
      json.add("network", inner.getJson(resources, dataSerializer));
      return json;
    }

    public void _free() {
      super._free();
    }

    public @Override @SuppressWarnings("unused") SubnetLeg addRef() {
      return (SubnetLeg) super.addRef();
    }

    public static @SuppressWarnings("unused") SubnetLeg[] addRefs(SubnetLeg[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(SubnetLeg::addRef)
          .toArray((x) -> new SubnetLeg[x]);
    }

  }

  public @Override @SuppressWarnings("unused") ImgLinearSubnetLayer addRef() {
    return (ImgLinearSubnetLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgLinearSubnetLayer[] addRefs(ImgLinearSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgLinearSubnetLayer::addRef)
        .toArray((x) -> new ImgLinearSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgLinearSubnetLayer[][] addRefs(ImgLinearSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgLinearSubnetLayer::addRefs)
        .toArray((x) -> new ImgLinearSubnetLayer[x][]);
  }
}
