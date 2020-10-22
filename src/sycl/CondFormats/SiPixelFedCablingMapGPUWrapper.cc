// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// SYCL includes
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

// CMSSW includes
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cablingMapHost = sycl::malloc_host<SiPixelFedCablingMapGPU>(1, dpct::get_default_queue());
  std::memcpy(cablingMapHost, &cablingMap, sizeof(SiPixelFedCablingMapGPU));
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {
  sycl::free(cablingMapHost, dpct::get_default_queue());
}

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(sycl::queue* cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(
      cudaStream,
      [this](GPUData& data, sycl::queue* stream) {
        // allocate
        data.cablingMapDevice = sycl::malloc_device<SiPixelFedCablingMapGPU>(1, dpct::get_default_queue());

        // transfer
        stream->memcpy(data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU));
      } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
      });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(sycl::queue* cudaStream) const {
  const auto& data = modToUnp_.dataForCurrentDeviceAsync(
      cudaStream,
      [this](ModulesToUnpack& data, sycl::queue* stream) {
        data.modToUnpDefault =
            (unsigned char*)sycl::malloc_device(pixelgpudetails::MAX_SIZE_BYTE_BOOL, dpct::get_default_queue());
        stream->memcpy(
            data.modToUnpDefault, this->modToUnpDefault.data(), this->modToUnpDefault.size() * sizeof(unsigned char));
      } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
      });
  return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() { sycl::free(cablingMapDevice, dpct::get_default_queue()); }

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() {
  sycl::free(modToUnpDefault, dpct::get_default_queue());
}
