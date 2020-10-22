#include <CL/sycl.hpp>

#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"
#include "SYCLCore/copyAsync.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

template <>
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  cms::sycltools::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(2001, stream);
  stream.memcpy(ret.get(), m_hitsModuleStart, 4 * 2001);
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float));
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t));
  return ret;
}

template <>
cms::sycltools::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int16_t[]>(2 * nHits(), stream);
  stream.memcpy(ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t));
  return ret;
}
