#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "SYCLCore/cudaCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  /*
  DPCT1003:176: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((gainForHLTonHost_ = sycl::malloc_host<SiPixelGainForHLTonGPU>(1, dpct::get_default_queue()), 0));
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  cudaCheck((sycl::free(gainForHLTonHost_, dpct::get_default_queue()), 0));
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cudaCheck((sycl::free(gainForHLTonGPU, dpct::get_default_queue()), 0));
  cudaCheck((sycl::free(gainDataOnGPU, dpct::get_default_queue()), 0));
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue *cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, sycl::queue *stream) {
  cudaCheck((data.gainForHLTonGPU = sycl::malloc_device<SiPixelGainForHLTonGPU>(1, dpct::get_default_queue()), 0));
  /*
    DPCT1003:177: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((data.gainDataOnGPU = (SiPixelGainForHLTonGPU_DecodingStructure*)sycl::malloc_device(
                 this->gainData_.size(), dpct::get_default_queue()),
             0));
  // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
  cudaCheck((stream->memcpy(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size()), 0));

  /*
    DPCT1003:178: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((stream->memcpy(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)), 0));
  /*
    DPCT1003:179: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((stream->memcpy(&(data.gainForHLTonGPU->v_pedestals),
                            &(data.gainDataOnGPU),
                            sizeof(SiPixelGainForHLTonGPU_DecodingStructure*)),
             0));
  }
  catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
  });
return data.gainForHLTonGPU;
}
