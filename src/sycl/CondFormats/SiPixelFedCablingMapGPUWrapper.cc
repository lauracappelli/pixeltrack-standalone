// C++ includes
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes

// CMSSW includes
#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cudaCheck((cablingMapHost = sycl::malloc_host<SiPixelFedCablingMapGPU>(1, dpct::get_default_queue()), 0));
  std::memcpy(cablingMapHost, &cablingMap, sizeof(SiPixelFedCablingMapGPU));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

/*
DPCT1003:211: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
*/
SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {
  cudaCheck((sycl::free(cablingMapHost, dpct::get_default_queue()), 0));
}

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(sycl::queue *cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, sycl::queue *stream) {
  // allocate
  cudaCheck((data.cablingMapDevice = sycl::malloc_device<SiPixelFedCablingMapGPU>(1, dpct::get_default_queue()), 0));

  // transfer
  cudaCheck((stream->memcpy(data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU)), 0));
  }
  catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
  });
return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(sycl::queue *cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, sycl::queue *stream) {
  cudaCheck((data.modToUnpDefault =
                 (unsigned char*)sycl::malloc_device(pixelgpudetails::MAX_SIZE_BYTE_BOOL, dpct::get_default_queue()),
             0));
  cudaCheck(
      (stream->memcpy(
           data.modToUnpDefault, this->modToUnpDefault.data(), this->modToUnpDefault.size() * sizeof(unsigned char)),
       0));
      }
      catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
      }
      });
return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() {
  cudaCheck((sycl::free(cablingMapDevice, dpct::get_default_queue()), 0));
}

/*
DPCT1003:212: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
*/
SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() {
  cudaCheck((sycl::free(modToUnpDefault, dpct::get_default_queue()), 0));
}
