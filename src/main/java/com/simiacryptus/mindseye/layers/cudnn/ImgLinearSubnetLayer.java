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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgLinearSubnetLayer extends LayerBase implements MultiPrecision {

  private static final Logger logger = LoggerFactory.getLogger(ImgLinearSubnetLayer.class);
  private final RefList<SubnetLeg> legs = new RefArrayList<>();
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private boolean parallel = true;

  public ImgLinearSubnetLayer() {
    super();
  }

  protected ImgLinearSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    setParallel(json.get("parallel").getAsBoolean());
    JsonArray jsonArray = json.get("legs").getAsJsonArray();
    for (int i = 0; i < jsonArray.size(); i++) {
      legs.add(new SubnetLeg(jsonArray.get(i).getAsJsonObject(), rs));
    }
  }

  @Nullable
  public RefList<SubnetLeg> getLegs() {
    return legs == null ? null : legs.addRef();
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(Precision precision) {
    this.precision = precision;
  }

  public boolean isParallel() {
    return parallel;
  }

  public void setParallel(boolean parallel) {
    this.parallel = parallel;
  }

  @Override
  public void setFrozen(final boolean frozen) {
    legs.stream().map(x -> {
      try {
        return x.inner.addRef();
      } finally {
        x.freeRef();
      }
    }).forEach(x -> {
      try {
        x.setFrozen(frozen);
      } finally {
        RefUtil.freeRef(x);
      }
    });
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgLinearSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgLinearSubnetLayer(json, rs);
  }

  public void add(int from, int to, @Nullable Layer layer) {
    RefList<SubnetLeg> temp_06_0010 = getLegs();
    assert temp_06_0010 != null;
    temp_06_0010.add(new SubnetLeg(layer == null ? null : layer.addRef(), from, to));
    temp_06_0010.freeRef();
    if (null != layer)
      layer.freeRef();
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int length = inputData.length();
    int maxBand = legs.stream().mapToInt(x -> {
      int temp_06_0003 = x.toBand;
      x.freeRef();
      return temp_06_0003;
    }).max().getAsInt();
    assert maxBand == inputDims[2] : maxBand + " != " + inputDims[2];
    assert RefIntStream.range(0, maxBand).allMatch(i -> 1 == legs.stream().filter(x -> {
      boolean temp_06_0004 = x.fromBand <= i && x.toBand > i;
      x.freeRef();
      return temp_06_0004;
    }).count());
    CudaTensor passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
      CudaTensor cudaTensor = new CudaTensor(gpu.allocate(inputData.getElements() * precision.size, MemoryType.Device, true),
          gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
      gpu.freeRef();
      return cudaTensor;
    }, inputData.addRef()));
    inputData.freeRef();
    AtomicInteger counter = new AtomicInteger(0);
    Result[] legResults = legs.stream()
        .map(RefUtil.wrapInterface((Function<? super SubnetLeg, ? extends Result>) leg -> {
          ImgBandSelectLayer imgBandSelectLayer = new ImgBandSelectLayer(leg.fromBand, leg.toBand);
          try {
            assert leg.inner != null;
            TensorList legData = Result.getData(imgBandSelectLayer.eval(input.addRef()));
            Result.Accumulator accumulator = new LegAccumulator(passback.addRef(), leg.addRef(), length, inputDims, counter, legs.addRef(), precision, input.getAccumulator());
            return leg.inner.eval(new Result(legData, accumulator));
          } finally {
            imgBandSelectLayer.freeRef();
            leg.freeRef();
          }
        }, passback, input))
        .toArray(i -> new Result[i]);
    SumInputsLayer sumInputsLayer = new SumInputsLayer();
    sumInputsLayer.setParallel(parallel);
    sumInputsLayer.setPrecision(precision);
    try {
      return sumInputsLayer.eval(legResults);
    } finally {
      sumInputsLayer.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    JsonArray jsonArray = new JsonArray();
    legs.stream().map(x -> {
      try {
        return x.getJson(resources, dataSerializer);
      } finally {
        x.freeRef();
      }
    }).forEach(element -> jsonArray.add(element));
    json.add("legs", jsonArray);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public void _free() {
    if (null != legs)
      legs.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgLinearSubnetLayer addRef() {
    return (ImgLinearSubnetLayer) super.addRef();
  }

  public static class SubnetLeg extends ReferenceCountingBase {

    @Nullable
    private final Layer inner;
    private final int fromBand;
    private final int toBand;

    public SubnetLeg(@Nullable final Layer inner, final int fromBand, final int toBand) {
      this.inner = inner;
      this.fromBand = fromBand;
      this.toBand = toBand;
    }

    protected SubnetLeg(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
      fromBand = json.getAsJsonPrimitive("fromBand").getAsInt();
      toBand = json.getAsJsonPrimitive("toBand").getAsInt();
      inner = Layer.fromJson(json.getAsJsonObject("network"), rs);
    }

    @Nonnull
    public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
      @Nonnull final JsonObject json = new JsonObject();
      json.addProperty("fromBand", fromBand);
      json.addProperty("toBand", toBand);
      assert inner != null;
      json.add("network", inner.getJson(resources, dataSerializer));
      return json;
    }

    public void _free() {
      if (null != inner)
        inner.freeRef();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    SubnetLeg addRef() {
      return (SubnetLeg) super.addRef();
    }
  }

  private static class LegAccumulator extends Result.Accumulator {

    private final CudaTensor passback;
    private final SubnetLeg leg;
    private final int length;
    private final int[] inputDims;
    private final AtomicInteger counter;
    private RefList<SubnetLeg> legs;
    private Precision precision;
    private Result.Accumulator accumulator;

    public LegAccumulator(CudaTensor passback, SubnetLeg leg, int length, int[] inputDims, AtomicInteger counter, RefList<SubnetLeg> legs, Precision precision, Result.Accumulator accumulator) {
      this.passback = passback;
      this.leg = leg;
      this.length = length;
      this.inputDims = inputDims;
      this.counter = counter;
      this.legs = legs;
      this.precision = precision;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nonnull TensorList delta) {
      int[] outputDimensions = delta.getDimensions();
      synchronized (passback) {
        CudaSystem.run(RefUtil.wrapInterface((RefConsumer<CudnnHandle>) gpu -> {
              final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision, length,
                  outputDimensions[2], outputDimensions[1], outputDimensions[0],
                  inputDims[2] * inputDims[1] * inputDims[0], inputDims[1] * inputDims[0], inputDims[0], 1);
              final int byteOffset = viewDescriptor.cStride * leg.fromBand * precision.size;
              assert delta.length() == length;
              assert passback.getDeviceId() == gpu.getDeviceId();
              @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta.addRef(), precision,
                  MemoryType.Device, true);
              @Nonnull final CudaMemory passbackBuffer = passback.getMemory(gpu.addRef());
              CudaMemory errorPtrMemory = deltaTensor.getMemory(gpu.addRef());
              assert passbackBuffer != null;
              passbackBuffer.synchronize();
              assert errorPtrMemory != null;
              gpu.cudnnTransformTensor(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                  errorPtrMemory.getPtr(), precision.getPointer(0.0), viewDescriptor.getPtr(),
                  passbackBuffer.getPtr().withByteOffset(byteOffset));
              gpu.freeRef();
              deltaTensor.freeRef();
              viewDescriptor.freeRef();
              errorPtrMemory.dirty();
              errorPtrMemory.freeRef();
              passbackBuffer.dirty();
              passbackBuffer.freeRef();
            }, passback.addRef(), delta.addRef(),
            leg.addRef()), passback.addRef());
      }
      delta.freeRef();
      if (counter.incrementAndGet() >= legs.size()) {
        counter.set(0);
        DeltaSet<UUID> buffer = ctx == null ? null : ctx.addRef();
        this.accumulator.accept(buffer, new CudaTensorList(passback.addRef(), length, inputDims, precision));
      }
      if (null != ctx)
        ctx.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      passback.freeRef();
      accumulator.freeRef();
      leg.freeRef();
      legs.freeRef();
    }
  }
}
