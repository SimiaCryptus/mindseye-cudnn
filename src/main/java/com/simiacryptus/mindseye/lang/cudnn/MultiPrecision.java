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
import com.simiacryptus.mindseye.network.DAGNetwork;

import javax.annotation.Nullable;

/**
 * The interface Multi precision.
 */
public interface MultiPrecision {
  /**
   * Gets precision.
   *
   * @return the precision
   */
  @Nullable
  Precision getPrecision();

  /**
   * Sets precision.
   *
   * @param precision the precision
   */
  void setPrecision(Precision precision);

  /**
   * Sets precision.
   *
   * @param network   the network
   * @param precision the precision
   */
  static void setPrecision(final Layer network, final Precision precision) {
    try {
      if (network instanceof DAGNetwork) {
        ((DAGNetwork) network).visitLayers(layer -> {
          if (layer instanceof MultiPrecision) {
            ((MultiPrecision) layer).setPrecision(precision);
          }
          if (null != layer)
            layer.freeRef();
        });
      } else if (network instanceof MultiPrecision) {
        ((MultiPrecision) network).setPrecision(precision);
      }
    } finally {
      network.freeRef();
    }
  }

}
