package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.LayerTestBase;

public abstract class CudnnLayerTestBase extends LayerTestBase {
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }
}
