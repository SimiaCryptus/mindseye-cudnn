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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefConsumer;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

@SuppressWarnings("serial")
public class ImgLinearSubnetLayer extends LayerBase implements MultiPrecision {

  private static final Logger logger = LoggerFactory.getLogger(ImgLinearSubnetLayer.class);
  private final RefList<SubnetLeg> legs = new RefArrayList<>();
  private Precision precision = CudaSettings.INSTANCE().defaultPrecision;
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

  @Nonnull
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
    ReferenceCounting.freeRefs(inObj);
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
    CudaTensor passback = CudaSystem.run(RefUtil.wrapInterface((Function<CudnnHandle, CudaTensor>) gpu -> {
      return new CudaTensor(gpu.allocate(inputData.getElements() * precision.size, MemoryType.Device, true),
          gpu.newTensorDescriptor(precision, length, inputDims[2], inputDims[1], inputDims[0]), precision);
    }, inputData.addRef()));
    inputData.freeRef();
    AtomicInteger counter = new AtomicInteger(0);
    Result[] legResults = legs.stream()
        .map(RefUtil.wrapInterface((Function<? super SubnetLeg, ? extends Result>) leg -> {
          ImgBandSelectLayer imgBandSelectLayer = new ImgBandSelectLayer(leg.fromBand, leg.toBand);
          Result temp_06_0011 = imgBandSelectLayer.eval(input.addRef());
          assert temp_06_0011 != null;
          TensorList legData = temp_06_0011.getData();
          temp_06_0011.freeRef();
          imgBandSelectLayer.freeRef();
          assert leg.inner != null;
          Result.Accumulator accumulator = new Result.Accumulator() {
            {
            }

            @Override
            public void accept(@Nullable DeltaSet<UUID> ctx, @Nonnull TensorList delta) {
              int[] outputDimensions = delta.getDimensions();
              synchronized (passback) {
                CudaSystem.run(RefUtil.wrapInterface((RefConsumer<CudnnHandle>) gpu -> {
                      @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(precision, length,
                          outputDimensions[2], outputDimensions[1], outputDimensions[0],
                          inputDims[2] * inputDims[1] * inputDims[0], inputDims[1] * inputDims[0], inputDims[0], 1);
                      final int byteOffset = viewDescriptor.cStride * leg.fromBand * precision.size;
                      assert delta.length() == length;
                      assert passback.getDeviceId() == gpu.getDeviceId();
                      @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta.addRef(), precision,
                          MemoryType.Device, true);
                      @Nonnull final CudaMemory passbackBuffer = passback.getMemory(gpu);
                      CudaMemory errorPtrMemory = deltaTensor.getMemory(gpu);
                      assert passbackBuffer != null;
                      passbackBuffer.synchronize();
                      assert errorPtrMemory != null;
                      gpu.cudnnTransformTensor(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(),
                          errorPtrMemory.getPtr(), precision.getPointer(0.0), viewDescriptor.getPtr(),
                          passbackBuffer.getPtr().withByteOffset(byteOffset));
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
                input.accumulate(ctx == null ? null : ctx.addRef(),
                    new CudaTensorList(passback.addRef(), length, inputDims, precision));
              }
              if (null != ctx)
                ctx.freeRef();
            }

            public @SuppressWarnings("unused")
            void _free() {
            }
          };
          Result temp_06_0006 = leg.inner.eval(new Result(legData, accumulator));
          leg.freeRef();
          return temp_06_0006;
        }, passback == null ? null : passback.addRef(), input.addRef()))
        .toArray(i -> new Result[i]);
    if (null != passback)
      passback.freeRef();
    input.freeRef();
    SumInputsLayer temp_06_0009 = new SumInputsLayer();
    temp_06_0009.setParallel(parallel);
    SumInputsLayer temp_06_0012 = temp_06_0009.addRef();
    temp_06_0012.setPrecision(precision);
    SumInputsLayer sumInputsLayer = RefUtil.addRef(temp_06_0012);
    temp_06_0012.freeRef();
    temp_06_0009.freeRef();
    Result temp_06_0005 = sumInputsLayer.eval(RefUtil.addRefs(legResults));
    sumInputsLayer.freeRef();
    ReferenceCounting.freeRefs(legResults);
    return temp_06_0005;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    JsonArray jsonArray = new JsonArray();
    legs.stream().map(x -> {
      JsonObject temp_06_0007 = x.getJson(resources, dataSerializer);
      x.freeRef();
      return temp_06_0007;
    }).forEach(jsonArray::add);
    json.add("legs", jsonArray);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  @Nonnull
  @Override
  public void setFrozen(final boolean frozen) {
    legs.stream().map(x -> {
      Layer temp_06_0008 = x.inner;
      x.freeRef();
      return temp_06_0008;
    }).forEach(x -> {
      assert x != null;
      x.setFrozen(frozen);
      x.freeRef();
    });
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
      Layer temp_06_0001 = inner == null ? null : inner.addRef();
      this.inner = temp_06_0001 == null ? null : temp_06_0001.addRef();
      if (null != temp_06_0001)
        temp_06_0001.freeRef();
      if (null != inner)
        inner.freeRef();
      this.fromBand = fromBand;
      this.toBand = toBand;
    }

    protected SubnetLeg(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
      fromBand = json.getAsJsonPrimitive("fromBand").getAsInt();
      toBand = json.getAsJsonPrimitive("toBand").getAsInt();
      Layer temp_06_0002 = Layer.fromJson(json.getAsJsonObject("network"), rs);
      inner = temp_06_0002.addRef();
      temp_06_0002.freeRef();
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
}
