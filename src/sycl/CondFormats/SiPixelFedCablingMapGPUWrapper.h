#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include <set>

#include <CL/sycl.hpp>

#include "SYCLCore/ESProduct.h"
#include "SYCLCore/CUDAHostAllocator.h"
#include "SYCLCore/device_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(sycl::queue *cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(sycl::queue *cudaStream) const;

private:
  std::vector<unsigned char, CUDAHostAllocator<unsigned char>> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU *cablingMapHost = nullptr;  // pointer to struct in CPU

  struct GPUData {
    ~GPUData();
    SiPixelFedCablingMapGPU *cablingMapDevice = nullptr;  // pointer to struct in GPU
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  };
  cms::sycltools::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
