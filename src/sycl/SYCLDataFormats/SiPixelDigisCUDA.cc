#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLDataFormats/SiPixelDigisCUDA.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/copyAsync.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, sycl::queue *stream) {
  xx_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::sycltools::make_device_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cms::sycltools::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::sycltools::make_device_unique<DeviceConstView>(stream);
  cms::sycltools::copyAsync(view_d, view, stream);
}

cms::sycltools::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(sycl::queue *stream) const {
  auto ret = cms::sycltools::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::sycltools::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(sycl::queue *stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::sycltools::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(sycl::queue *stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::sycltools::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(sycl::queue *stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::sycltools::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
