#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"
#include "SYCLCore/copyAsync.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

template <>
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  cms::sycltools::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(2001, stream);
  cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  cudaMemcpyAsync(ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), cudaMemcpyDefault, stream);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nHits(), stream);
  cudaMemcpyAsync(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), cudaMemcpyDefault, stream);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::sycltools::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaMemcpyAsync(ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), cudaMemcpyDefault, stream);
  return ret;
}
