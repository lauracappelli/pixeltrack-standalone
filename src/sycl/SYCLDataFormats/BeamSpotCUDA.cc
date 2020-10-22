#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLDataFormats/BeamSpotCUDA.h"

#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, sycl::queue* stream) {
  data_d_ = cms::sycltools::make_device_unique<Data>(stream);
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(data_d_.get(), data_h, sizeof(Data)), 0));
}
