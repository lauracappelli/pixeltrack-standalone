#include <CL/sycl.hpp>

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLDataFormats/BeamSpotCUDA.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, sycl::queue* stream) {
  data_d_ = cms::sycltools::make_device_unique<Data>(stream);
  stream->memcpy(data_d_.get(), data_h, sizeof(Data));
}
