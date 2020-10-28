#include <iostream>
#include <fstream>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "Geometry/phase1PixelTopology.h"
#include "CondFormats/PixelCPEFast.h"

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

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(sycl::queue stream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData &data, sycl::queue stream) {
    // and now copy to device...
    data.h_paramsOnGPU.m_commonParams =
        sycl::malloc_device<pixelCPEforGPU::CommonParams>(sizeof(pixelCPEforGPU::CommonParams), stream);
    data.h_paramsOnGPU.m_detParams = sycl::malloc_device<pixelCPEforGPU::DetParams>(
        this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams), stream);
    data.h_paramsOnGPU.m_averageGeometry =
        sycl::malloc_device<phase1PixelTopology::AverageGeometry>(sizeof(pixelCPEforGPU::AverageGeometry), stream);
    data.h_paramsOnGPU.m_layerGeometry =
        sycl::malloc_device<pixelCPEforGPU::LayerGeometry>(sizeof(pixelCPEforGPU::LayerGeometry), stream);
    data.d_paramsOnGPU = sycl::malloc_device<pixelCPEforGPU::ParamsOnGPU>(1, stream);

    stream.memcpy(data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU));
    stream.memcpy(
        (void *)data.h_paramsOnGPU.m_commonParams, &this->m_commonParamsGPU, sizeof(pixelCPEforGPU::CommonParams));
    stream.memcpy((void *)data.h_paramsOnGPU.m_averageGeometry,
                  &this->m_averageGeometry,
                  sizeof(pixelCPEforGPU::AverageGeometry));
    stream.memcpy(
        (void *)data.h_paramsOnGPU.m_layerGeometry, &this->m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry));
    stream.memcpy((void *)data.h_paramsOnGPU.m_detParams,
                  this->m_detParamsGPU.data(),
                  this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams));
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
