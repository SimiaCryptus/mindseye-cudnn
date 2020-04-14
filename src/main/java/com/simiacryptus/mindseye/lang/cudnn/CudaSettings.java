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
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.LocalAppSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.File;
import java.util.HashMap;

import static com.simiacryptus.lang.Settings.get;

/**
 * The type Cuda settings.
 */
public class CudaSettings implements Settings {

  private static final Logger logger = LoggerFactory.getLogger(CudaSettings.class);

  @Nullable
  private static transient CudaSettings INSTANCE = null;

  /**
   * The Default devices.
   */
  public final String defaultDevices;
  /**
   * The Memory cache mode.
   */
  public final PersistanceMode memoryCacheMode = get("CUDA_CACHE_MODE", PersistanceMode.WEAK);
  /**
   * The All dense.
   */
  public final boolean allDense = false;
  /**
   * The Verbose.
   */
  public final boolean verbose = false;
  /**
   * The Async free load threshold.
   */
  public final double asyncFreeLoadThreshold = 0.5;
  /**
   * The Max total memory.
   */
  public final long maxTotalMemory = get("MAX_TOTAL_MEMORY", 12 * CudaMemory.GiB);
  /**
   * The Max alloc size.
   */
  public final long maxAllocSize = (long) get("MAX_ALLOC_SIZE",
      (double) Precision.Double.size * (Integer.MAX_VALUE / 2 - 1L));
  /**
   * The Max io elements.
   */
  public final double maxIoElements = get("MAX_IO_ELEMENTS",
      (double) 256 * CudaMemory.MiB);
  /**
   * The Convolution workspace size limit.
   */
  public final long convolutionWorkspaceSizeLimit = (long) get("CONVOLUTION_WORKSPACE_SIZE_LIMIT",
      (double) 126 * CudaMemory.MiB);
  /**
   * The Disable.
   */
  public final boolean disable = get("DISABLE_CUDNN", false);
  /**
   * The Force single gpu.
   */
  public final boolean forceSingleGpu = get("FORCE_SINGLE_GPU", true);
  /**
   * The Max filter elements.
   */
  public final long maxFilterElements = (long) get("MAX_FILTER_ELEMENTS",
      (double) 256 * CudaMemory.MiB);
  /**
   * The Conv para 2.
   */
  public final boolean conv_para_2 = get("CONV_PARA_2", false);
  /**
   * The Conv para 1.
   */
  public final boolean conv_para_1 = get("CONV_PARA_1", true);
  /**
   * The Conv para 3.
   */
  public final boolean conv_para_3 = get("CONV_PARA_3", false);
  /**
   * The Max device memory.
   */
  public final long maxDeviceMemory = get("MAX_DEVICE_MEMORY", 8 * CudaMemory.GiB);
  /**
   * The Log stack.
   */
  public final boolean logStack = get("CUDA_LOG_STACK", false);
  /**
   * The Profile memory io.
   */
  public final boolean profileMemoryIO = get("CUDA_PROFILE_MEM_IO", false);
  /**
   * The Enable managed.
   */
  public final boolean enableManaged = get("CUDA_MANAGED_MEM", false);
  /**
   * The Sync before free.
   */
  public final boolean syncBeforeFree = get("SYNC_BEFORE_FREE", false);
  /**
   * The Memory cache ttl.
   */
  public final int memoryCacheTTL = get("CUDA_CACHE_TTL", 5);
  /**
   * The Convolution cache.
   */
  public final boolean convolutionCache = true;
  /**
   * The Handles per device.
   */
  public final int handlesPerDevice = get("CUDA_HANDLES_PER_DEVICE", 8);
  private Precision defaultPrecision = get("CUDA_DEFAULT_PRECISION", Precision.Double);

  private CudaSettings() {
    CudaSystem.printHeader(System.out);
    HashMap<String, String> appSettings = LocalAppSettings.read();
    String spark_home = System.getenv("SPARK_HOME");
    File sparkHomeFile = new File(spark_home == null ? "." : spark_home);
    if (sparkHomeFile.exists()) {
      assert appSettings != null;
      appSettings.putAll(LocalAppSettings.read(sparkHomeFile));
    }
    assert appSettings != null;
    if (appSettings.containsKey("worker.index"))
      System.setProperty("CUDA_DEVICES", appSettings.get("worker.index"));
    defaultDevices = get("CUDA_DEVICES", "");
  }

  /**
   * Gets default precision.
   *
   * @return the default precision
   */
  public Precision getDefaultPrecision() {
    return defaultPrecision;
  }

  /**
   * Sets default precision.
   *
   * @param defaultPrecision the default precision
   */
  public void setDefaultPrecision(Precision defaultPrecision) {
    this.defaultPrecision = defaultPrecision;
  }

  /**
   * Instance cuda settings.
   *
   * @return the cuda settings
   */
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
