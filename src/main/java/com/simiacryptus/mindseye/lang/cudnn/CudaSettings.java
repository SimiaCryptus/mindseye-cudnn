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

import com.simiacryptus.lang.Settings;
import com.simiacryptus.ref.lang.PersistanceMode;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.LocalAppSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;

public class CudaSettings implements Settings {

  private static final Logger logger = LoggerFactory.getLogger(CudaSettings.class);

  @Nullable
  private static transient CudaSettings INSTANCE = null;
  public final String defaultDevices;
  @Nonnull
  public final PersistanceMode memoryCacheMode;
  public final boolean allDense;
  public final boolean verbose;
  public final double asyncFreeLoadThreshold = 0.5;
  private final long maxTotalMemory;
  private final long maxAllocSize;
  private final double maxIoElements;
  private final long convolutionWorkspaceSizeLimit;
  private final boolean disable;
  private final boolean forceSingleGpu;
  private final long maxFilterElements;
  private final boolean conv_para_2;
  private final boolean conv_para_1;
  private final boolean conv_para_3;
  private final long maxDeviceMemory;
  private final boolean logStack;
  private final boolean profileMemoryIO;
  private final boolean enableManaged;
  private final boolean syncBeforeFree;
  private final int memoryCacheTTL;
  private final boolean convolutionCache;
  private final int handlesPerDevice;
  public Precision defaultPrecision;

  private CudaSettings() {
    RefHashMap<String, String> appSettings = LocalAppSettings.read();
    String spark_home = RefSystem.getenv("SPARK_HOME");
    File sparkHomeFile = new File(spark_home == null ? "." : spark_home);
    if (sparkHomeFile.exists()) {
      assert appSettings != null;
      appSettings.putAll(LocalAppSettings.read(sparkHomeFile));
    }
    assert appSettings != null;
    if (appSettings.containsKey("worker.index"))
      RefSystem.setProperty("CUDA_DEVICES", appSettings.get("worker.index"));
    appSettings.freeRef();
    maxTotalMemory = Settings.get("MAX_TOTAL_MEMORY", 12 * CudaMemory.GiB);
    maxDeviceMemory = Settings.get("MAX_DEVICE_MEMORY", 6 * CudaMemory.GiB);
    maxAllocSize = (long) Settings.get("MAX_ALLOC_SIZE", (double) Precision.Double.size * (Integer.MAX_VALUE / 2 - 1L));
    maxFilterElements = (long) Settings.get("MAX_FILTER_ELEMENTS", (double) 126 * CudaMemory.MiB);
    maxIoElements = Settings.get("MAX_IO_ELEMENTS", (double) 126 * CudaMemory.MiB);
    convolutionWorkspaceSizeLimit = (long) Settings.get("CONVOLUTION_WORKSPACE_SIZE_LIMIT",
        (double) 126 * CudaMemory.MiB);
    disable = Settings.get("DISABLE_CUDNN", false);
    forceSingleGpu = Settings.get("FORCE_SINGLE_GPU", true);
    conv_para_1 = Settings.get("CONV_PARA_1", false);
    conv_para_2 = Settings.get("CONV_PARA_2", false);
    conv_para_3 = Settings.get("CONV_PARA_3", false);
    memoryCacheMode = Settings.get("CUDA_CACHE_MODE", PersistanceMode.WEAK);
    logStack = Settings.get("CUDA_LOG_STACK", false);
    profileMemoryIO = Settings.get("CUDA_PROFILE_MEM_IO", false);
    enableManaged = false;
    syncBeforeFree = false;
    memoryCacheTTL = 5;
    convolutionCache = true;
    defaultDevices = Settings.get("CUDA_DEVICES", "");
    this.handlesPerDevice = Settings.get("CUDA_HANDLES_PER_DEVICE", 8);
    defaultPrecision = Precision.valueOf(Settings.get("CUDA_DEFAULT_PRECISION", Precision.Float.name()));
    allDense = false;
    verbose = false;
  }

  public long getConvolutionWorkspaceSizeLimit() {
    return convolutionWorkspaceSizeLimit;
  }

  public int getHandlesPerDevice() {
    return handlesPerDevice;
  }

  public double getMaxAllocSize() {
    return maxAllocSize;
  }

  public double getMaxDeviceMemory() {
    return maxDeviceMemory;
  }

  public long getMaxFilterElements() {
    return maxFilterElements;
  }

  public double getMaxIoElements() {
    return maxIoElements;
  }

  public double getMaxTotalMemory() {
    return maxTotalMemory;
  }

  public int getMemoryCacheTTL() {
    return memoryCacheTTL;
  }

  public boolean isConv_para_1() {
    return conv_para_1;
  }

  public boolean isConv_para_2() {
    return conv_para_2;
  }

  public boolean isConv_para_3() {
    return conv_para_3;
  }

  public boolean isConvolutionCache() {
    return convolutionCache;
  }

  public boolean isDisable() {
    return disable;
  }

  public boolean isEnableManaged() {
    return enableManaged;
  }

  public boolean isForceSingleGpu() {
    return forceSingleGpu;
  }

  public boolean isLogStack() {
    return logStack;
  }

  public boolean isProfileMemoryIO() {
    return profileMemoryIO;
  }

  public boolean isSyncBeforeFree() {
    return syncBeforeFree;
  }

  @Nullable
  public static CudaSettings INSTANCE() {
    if (null == INSTANCE) {
      synchronized (CudaSettings.class) {
        if (null == INSTANCE) {
          INSTANCE = new CudaSettings();
          logger.info(
              RefString.format("Initialized %s = %s", INSTANCE.getClass().getSimpleName(), JsonUtil.toJson(INSTANCE)));
        }
      }
    }
    return INSTANCE;
  }

}
