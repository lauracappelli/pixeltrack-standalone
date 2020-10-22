#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include <CL/sycl.hpp>

#include "SYCLDataFormats/BeamSpotCUDA.h"
#include "SYCLDataFormats/SiPixelClustersCUDA.h"
#include "SYCLDataFormats/SiPixelDigisCUDA.h"
#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"

namespace pixelgpudetails {

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel() = default;
    ~PixelRecHitGPUKernel() = default;

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    TrackingRecHit2DCUDA makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                       SiPixelClustersCUDA const& clusters_d,
                                       BeamSpotCUDA const& bs_d,
                                       pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                       sycl::queue stream) const;
  };
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
