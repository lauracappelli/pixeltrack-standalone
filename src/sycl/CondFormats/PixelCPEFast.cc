#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <fstream>

#include "Geometry/phase1PixelTopology.h"
#include "SYCLCore/cudaCheck.h"
#include "CondFormats/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  constexpr float micronsToCm = 1.0e-4;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(&m_commonParamsGPU), sizeof(pixelCPEforGPU::CommonParams));
    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    m_detParamsGPU.resize(ndetParams);
    in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(&m_averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(&m_layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  cpuData_ = {
      &m_commonParamsGPU,
      m_detParamsGPU.data(),
      &m_layerGeometry,
      &m_averageGeometry,
  };
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(sycl::queue *cudaStream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData &data, sycl::queue *stream) {
  // and now copy to device...
  cudaCheck((data.h_paramsOnGPU.m_commonParams = (const pixelCPEforGPU::CommonParams *)sycl::malloc_device(
                 sizeof(pixelCPEforGPU::CommonParams), dpct::get_default_queue()),
             0));
  /*
    DPCT1003:213: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((data.h_paramsOnGPU.m_detParams = (const pixelCPEforGPU::DetParams *)sycl::malloc_device(
                 this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams), dpct::get_default_queue()),
             0));
  /*
    DPCT1003:214: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((data.h_paramsOnGPU.m_averageGeometry = (const phase1PixelTopology::AverageGeometry *)sycl::malloc_device(
                 sizeof(pixelCPEforGPU::AverageGeometry), dpct::get_default_queue()),
             0));
  /*
    DPCT1003:215: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((data.h_paramsOnGPU.m_layerGeometry = (const pixelCPEforGPU::LayerGeometry *)sycl::malloc_device(
                 sizeof(pixelCPEforGPU::LayerGeometry), dpct::get_default_queue()),
             0));
  /*
    DPCT1003:216: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((data.d_paramsOnGPU = sycl::malloc_device<pixelCPEforGPU::ParamsOnGPU>(1, dpct::get_default_queue()), 0));

  cudaCheck((stream->memcpy(data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)), 0));
  cudaCheck(
      (stream->memcpy(
           (void *)data.h_paramsOnGPU.m_commonParams, &this->m_commonParamsGPU, sizeof(pixelCPEforGPU::CommonParams)),
       0));
  /*
    DPCT1003:217: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((stream->memcpy((void *)data.h_paramsOnGPU.m_averageGeometry,
                            &this->m_averageGeometry,
                            sizeof(pixelCPEforGPU::AverageGeometry)),
             0));
  /*
    DPCT1003:218: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck(
      (stream->memcpy(
           (void *)data.h_paramsOnGPU.m_layerGeometry, &this->m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)),
       0));
  /*
    DPCT1003:219: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
  cudaCheck((stream->memcpy((void *)data.h_paramsOnGPU.m_detParams,
                            this->m_detParamsGPU.data(),
                            this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams)),
             0));
  }
  catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
  });
return data.d_paramsOnGPU;
}

PixelCPEFast::GPUData::~GPUData() {
  if (d_paramsOnGPU != nullptr) {
    sycl::free((void *)h_paramsOnGPU.m_commonParams, dpct::get_default_queue());
    sycl::free((void *)h_paramsOnGPU.m_detParams, dpct::get_default_queue());
    sycl::free((void *)h_paramsOnGPU.m_averageGeometry, dpct::get_default_queue());
    sycl::free(d_paramsOnGPU, dpct::get_default_queue());
  }
}
