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
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Binary sum layer.
 */
@SuppressWarnings("serial")
public class BinarySumLayer extends LayerBase implements MultiPrecision {

  private double leftFactor;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();
  private double rightFactor;

  /**
   * Instantiates a new Binary sum layer.
   */
  public BinarySumLayer() {
    this(1.0, 1.0);
  }

  /**
   * Instantiates a new Binary sum layer.
   *
   * @param leftFactor  the left factor
   * @param rightFactor the right factor
   */
  public BinarySumLayer(final double leftFactor, final double rightFactor) {
    this.leftFactor = leftFactor;
    this.rightFactor = rightFactor;
    freeze();
  }

  /**
   * Instantiates a new Binary sum layer.
   *
   * @param json the json
   */
  protected BinarySumLayer(@Nonnull final JsonObject json) {
    super(json);
    rightFactor = json.get("rightFactor").getAsDouble();
    leftFactor = json.get("leftFactor").getAsDouble();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    @Nonnull
    LinearActivationLayer left = new LinearActivationLayer();
    left.setScale(this.leftFactor);
    left.freeze();
    LinearActivationLayer right = new LinearActivationLayer();
    right.setScale(this.rightFactor);
    right.freeze();
    PipelineNetwork network = new PipelineNetwork(2);
    RefUtil.freeRef(network.add(new SumInputsLayer(),
        network.add(left, network.getInput(0)),
        network.add(right, network.getInput(1))
    ));
    return network;
  }

  /**
   * Gets left factor.
   *
   * @return the left factor
   */
  public double getLeftFactor() {
    return leftFactor;
  }

  /**
   * Sets left factor.
   *
   * @param leftFactor the left factor
   */
  public void setLeftFactor(final double leftFactor) {
    this.leftFactor = leftFactor;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  /**
   * Gets right factor.
   *
   * @return the right factor
   */
  public double getRightFactor() {
    return rightFactor;
  }

  /**
   * Sets right factor.
   *
   * @param rightFactor the right factor
   */
  public void setRightFactor(final double rightFactor) {
    this.rightFactor = rightFactor;
  }

  /**
   * From json binary sum layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the binary sum layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static BinarySumLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BinarySumLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result left = inObj[0].addRef();
    int inLength = inObj.length;
    if (inLength == 1) {
      if (rightFactor != 1) {
        RefUtil.freeRef(inObj);
        left.freeRef();
        throw new IllegalStateException();
      }
      if (leftFactor != 1) {
        RefUtil.freeRef(inObj);
        left.freeRef();
        throw new IllegalStateException();
      }
      RefUtil.freeRef(inObj);
      return left;
    }
    if (inLength > 2) {
      if (rightFactor != 1) {
        RefUtil.freeRef(inObj);
        left.freeRef();
        throw new IllegalStateException();
      }
      if (leftFactor != 1) {
        RefUtil.freeRef(inObj);
        left.freeRef();
        throw new IllegalStateException();
      }
      left.freeRef();
      return RefUtil.get(RefArrays.stream(inObj).reduce((a, b) -> {
        return eval(a, b);
      }));
    }
    assert inLength == 2;
    final TensorList leftData = left.getData();
    final Result right = inObj[1].addRef();
    RefUtil.freeRef(inObj);
    final TensorList rightData = right.getData();
    int[] leftDimensions = leftData.getDimensions();
    if (3 < leftDimensions.length) {
      leftData.freeRef();
      rightData.freeRef();
      left.freeRef();
      right.freeRef();
      throw new IllegalArgumentException("dimensions=" + RefArrays.toString(leftDimensions));
    }
    @Nonnull final int[] outputDimensions = {
        leftDimensions.length < 1 ? 0 : leftDimensions[0],
        leftDimensions.length < 2 ? 1 : leftDimensions[1],
        leftDimensions.length < 3 ? 1 : leftDimensions[2]
    };
    final int length = leftData.length();
    int[] rightDimensions = rightData.getDimensions();
    if (length != rightData.length()) {
      leftData.freeRef();
      rightData.freeRef();
      left.freeRef();
      right.freeRef();
      throw new IllegalArgumentException();
    }
    if (Tensor.length(leftDimensions) != Tensor.length(rightDimensions)) {
      leftData.freeRef();
      rightData.freeRef();
      left.freeRef();
      right.freeRef();
      throw new IllegalArgumentException(
          RefArrays.toString(leftDimensions) + " != " + RefArrays.toString(rightDimensions));
    }
    if (!CudaSystem.isEnabled()) {
      leftData.freeRef();
      rightData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(left, right);
      compatibilityLayer.freeRef();
      return result;
    }
    boolean alive = left.isAlive() || right.isAlive();
    Result.Accumulator accumulator = new Accumulator(outputDimensions, length, precision, leftFactor, rightFactor, left.getAccumulator(), right.getAccumulator(), left.isAlive(), right.isAlive());
    right.freeRef();
    left.freeRef();
    CudaTensorList data = forwardEval(leftData, rightData, outputDimensions, length);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("rightFactor", rightFactor);
    json.addProperty("leftFactor", leftFactor);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BinarySumLayer addRef() {
    return (BinarySumLayer) super.addRef();
  }

  @NotNull
  private CudaTensorList forwardEval(TensorList leftData, TensorList rightData, int[] dimensions, int length) {
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu
              .newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
          final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
              dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0],
              dimensions[1] * dimensions[0], dimensions[0], 1);
          @Nullable final CudaTensor lPtr = gpu.getTensor(leftData.addRef(), precision,
              MemoryType.Device, false);
          @Nullable final CudaTensor rPtr = gpu.getTensor(rightData.addRef(), precision,
              MemoryType.Device, false);
          @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * Tensor.length(dimensions) * length,
              MemoryType.Managed.ifEnabled(), true);
          CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
          CudaMemory rPtrMemory = rPtr.getMemory(gpu.addRef());
          assert rPtrMemory != null;
          assert lPtrMemory != null;
          gpu.cudnnOpTensor(opDescriptor.getPtr(), precision.getPointer(leftFactor), lPtr.descriptor.getPtr(),
              lPtrMemory.getPtr(), precision.getPointer(rightFactor), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
              precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
          rPtr.freeRef();
          lPtr.freeRef();
          opDescriptor.freeRef();
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          gpu.freeRef();
          lPtrMemory.dirty();
          lPtrMemory.freeRef();
          rPtrMemory.dirty();
          rPtrMemory.freeRef();
          outputPtr.dirty();
          return new CudaTensorList(
              new CudaTensor(outputPtr, outputDescriptor, precision),
              length, dimensions, precision);
        },
        rightData.addRef(),
        leftData.addRef()
    ), leftData, rightData);
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] dimensions;
    private final int length;
    private final Precision precision;
    private final double leftFactor;
    private final double rightFactor;
    private Result.Accumulator leftAccumulator;
    private Result.Accumulator rightAccumulator;
    private boolean leftAlive;
    private boolean rightAlive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param dimensions       the dimensions
     * @param length           the length
     * @param precision        the precision
     * @param leftFactor       the left factor
     * @param rightFactor      the right factor
     * @param leftAccumulator  the left accumulator
     * @param rightAccumulator the right accumulator
     * @param leftAlive        the left alive
     * @param rightAlive       the right alive
     */
    public Accumulator(int[] dimensions, int length, Precision precision, double leftFactor, double rightFactor, Result.Accumulator leftAccumulator, Result.Accumulator rightAccumulator, boolean leftAlive, boolean rightAlive) {
      this.dimensions = dimensions;
      this.length = length;
      this.precision = precision;
      this.leftFactor = leftFactor;
      this.rightFactor = rightFactor;
      this.leftAccumulator = leftAccumulator;
      this.rightAccumulator = rightAccumulator;
      this.leftAlive = leftAlive;
      this.rightAlive = rightAlive;
    }


    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      Runnable leftRunnable = RefUtil.wrapInterface(() -> {
            if (leftAlive) {
              CudaTensorList tensorList = CudaSystem
                  .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                    @Nullable final CudaTensor lPtr = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                        MemoryType.Device, false);
                    @Nonnull final CudaMemory passbackPtr = gpu.allocate(
                        precision.size * Tensor.length(dimensions) * length, MemoryType.Managed.ifEnabled(),
                        true);
                    final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                        precision, length, dimensions[2], dimensions[1], dimensions[0],
                        dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                        dimensions[0], 1);
                    CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
                    assert lPtrMemory != null;
                    gpu.cudnnTransformTensor(precision.getPointer(leftFactor), lPtr.descriptor.getPtr(),
                        lPtrMemory.getPtr(), precision.getPointer(0.0), passbackDescriptor.getPtr(),
                        passbackPtr.getPtr());
                    lPtr.freeRef();
                    assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                    gpu.freeRef();
                    passbackPtr.dirty();
                    lPtrMemory.dirty();
                    lPtrMemory.freeRef();
                    return new CudaTensorList(
                        new CudaTensor(passbackPtr, passbackDescriptor, precision),
                        length, dimensions, precision);
                  }, delta == null ? null : delta.addRef()), delta == null ? null : delta.addRef());
              DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
              leftAccumulator.accept(buffer1, tensorList);
            }
          },
          delta == null ? null : delta.addRef(),
          buffer == null ? null : buffer.addRef(),
          leftAccumulator.addRef()
      );
      Runnable rightRunnable = RefUtil.wrapInterface(() -> {
            if (rightAlive) {
              CudaTensorList tensorList = CudaSystem
                  .run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
                        @Nullable final CudaTensor lPtr = gpu.getTensor(delta == null ? null : delta.addRef(), precision,
                            MemoryType.Device, false);
                        @Nonnull final CudaMemory outputPtr = gpu.allocate(
                            precision.size * Tensor.length(dimensions) * length, MemoryType.Managed.ifEnabled(),
                            true);
                        final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                            precision, length, dimensions[2], dimensions[1], dimensions[0],
                            dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0],
                            dimensions[0], 1);
                        CudaMemory lPtrMemory = lPtr.getMemory(gpu.addRef());
                        assert lPtrMemory != null;
                        gpu.cudnnTransformTensor(precision.getPointer(rightFactor), lPtr.descriptor.getPtr(),
                            lPtrMemory.getPtr(), precision.getPointer(0.0), passbackDescriptor.getPtr(),
                            outputPtr.getPtr());
                        gpu.freeRef();
                        lPtr.freeRef();
                        outputPtr.dirty();
                        lPtrMemory.dirty();
                        lPtrMemory.freeRef();
                        return new CudaTensorList(
                            new CudaTensor(outputPtr, passbackDescriptor, precision),
                            length, dimensions, precision);
                      }, delta == null ? null : delta.addRef()
                  ), delta == null ? null : delta.addRef());
              DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
              rightAccumulator.accept(buffer1, tensorList);
            }
          },
          buffer,
          delta,
          rightAccumulator.addRef()
      );
      if (CoreSettings.INSTANCE().singleThreaded)
        Util.runAllSerial(leftRunnable, rightRunnable);
      else
        Util.runAllParallel(leftRunnable, rightRunnable);
    }

    public @SuppressWarnings("unused")
    void _free() {
      leftAccumulator.freeRef();
      rightAccumulator.freeRef();
      super._free();
    }
  }
}
