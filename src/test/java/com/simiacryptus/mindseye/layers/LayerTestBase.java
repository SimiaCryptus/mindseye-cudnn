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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.test.unit.StandardLayerTests;
import com.simiacryptus.ref.lang.RefAware;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

public abstract @RefAware
class LayerTestBase extends StandardLayerTests {

  //  @Test(timeout = 15 * 60 * 1000)
  //  public void testMonteCarlo() throws Throwable {
  //    apply(this::monteCarlo);
  //  }

  public static @SuppressWarnings("unused")
  LayerTestBase[] addRefs(LayerTestBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerTestBase::addRef)
        .toArray((x) -> new LayerTestBase[x]);
  }

  public static @SuppressWarnings("unused")
  LayerTestBase[][] addRefs(LayerTestBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerTestBase::addRefs)
        .toArray((x) -> new LayerTestBase[x][]);
  }

  @Test(timeout = 15 * 60 * 1000)
  public void test() throws Throwable {
    run(this::run);
  }

  @Before
  public void setup() {
    reportingFolder = "reports/_reports";
    //GpuController.remove();
  }

  @After
  public void cleanup() {
    System.gc();
    //GpuController.remove();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LayerTestBase addRef() {
    return (LayerTestBase) super.addRef();
  }

}
