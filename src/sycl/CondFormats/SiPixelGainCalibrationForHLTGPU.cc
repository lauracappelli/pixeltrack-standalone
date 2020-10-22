#include <vector>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  gainForHLTonHost_ = sycl::malloc_host<SiPixelGainForHLTonGPU>(1, dpct::get_default_queue());
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  sycl::free(gainForHLTonHost_, dpct::get_default_queue());
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  sycl::free(gainForHLTonGPU, dpct::get_default_queue());
  sycl::free(gainDataOnGPU, dpct::get_default_queue());
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue* cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(
      cudaStream,
      [this](GPUData& data, sycl::queue* stream) {
        data.gainForHLTonGPU = sycl::malloc_device<SiPixelGainForHLTonGPU>(1, dpct::get_default_queue());
        data.gainDataOnGPU = (SiPixelGainForHLTonGPU_DecodingStructure*)sycl::malloc_device(this->gainData_.size(),
                                                                                            dpct::get_default_queue());
        // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
        stream->memcpy(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size());
        stream->memcpy(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU));
        stream->memcpy(&(data.gainForHLTonGPU->v_pedestals),
                       &(data.gainDataOnGPU),
                       sizeof(SiPixelGainForHLTonGPU_DecodingStructure*));
      } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
      });
  return data.gainForHLTonGPU;
}
