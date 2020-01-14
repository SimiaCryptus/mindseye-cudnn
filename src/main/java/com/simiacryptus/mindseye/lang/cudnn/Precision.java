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

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.mindseye.lang.Layer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnDataType;

import javax.annotation.Nonnull;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

public enum Precision {
  Double(cudnnDataType.CUDNN_DATA_DOUBLE, Sizeof.DOUBLE),
  Float(cudnnDataType.CUDNN_DATA_FLOAT, Sizeof.FLOAT);

  public final int code;
  public final int size;

  Precision(final int code, final int size) {
    this.code = code;
    this.size = size;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }

  public static double[] getDoubles(@Nonnull final float[] data) {
    return copy(data, new double[data.length]);
  }

  public static double[] copy(@Nonnull float[] from, double[] doubles) {
    for (int i = 0; i < from.length; i++) {
      doubles[i] = from[i];
    }
    return doubles;
  }

  public static float[] getFloats(@Nonnull final double[] data) {
    return copy(data, new float[data.length]);
  }

  public static float[] copy(@Nonnull double[] from, float[] to) {
    for (int i = 0; i < from.length; i++) {
      to[i] = (float) from[i];
    }
    return to;
  }

  public static void copy(@Nonnull double[] from, @Nonnull byte[] to, Precision precision) {
    if (precision == Float) copyFloats(from, to);
    else if (precision == Double) copyDoubles(from, to);
    else throw new RuntimeException();
  }

  public static void copy(@Nonnull byte[] from, @Nonnull double[] to, Precision precision) {
    if (precision == Float) copyFloats(from, to);
    else if (precision == Double) copyDoubles(from, to);
    else throw new RuntimeException();
  }

  public static void copyDoubles(@Nonnull double[] from, @Nonnull byte[] to) {
    @Nonnull DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    @Nonnull DoubleBuffer outBuffer = ByteBuffer.wrap(to).asDoubleBuffer();
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }

  public static void copyDoubles(@Nonnull byte[] from, @Nonnull double[] to) {
    @Nonnull DoubleBuffer inBuffer = ByteBuffer.wrap(from).asDoubleBuffer();
    @Nonnull DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }

  public static void copyFloats(@Nonnull double[] from, @Nonnull byte[] to) {
    @Nonnull DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    @Nonnull FloatBuffer outBuffer = ByteBuffer.wrap(to).asFloatBuffer();
    while (inBuffer.hasRemaining()) {
      outBuffer.put((float) inBuffer.get());
    }
  }

  public static void copyFloats(@Nonnull byte[] from, @Nonnull double[] to) {
    @Nonnull FloatBuffer inBuffer = ByteBuffer.wrap(from).asFloatBuffer();
    @Nonnull DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }

  @Nonnull
  public CudaPointer getPointer(@Nonnull final double... data) {
    switch (this) {
      case Float:
        return CudaPointer.to(Precision.getFloats(data));
      case Double:
        return CudaPointer.to(data);
      default:
        throw new IllegalStateException();
    }
  }

  @Nonnull
  public CudaPointer getPointer(@Nonnull final float... data) {
    switch (this) {
      case Float:
        return CudaPointer.to(data);
      case Double:
        return CudaPointer.to(Precision.getDoubles(data));
      default:
        throw new IllegalStateException();
    }
  }

}
