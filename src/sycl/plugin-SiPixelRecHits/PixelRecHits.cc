// C++ headers
#include <algorithm>
#include <numeric>

// SYCL headers
#include <CL/sycl.hpp>

// CMSSW headers
#include "SYCLCore/device_unique_ptr.h"
#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"        // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                         uint32_t* hitsLayerStart,
                         sycl::nd_item<3> item,
                         sycl::stream out) {
    auto i = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);

    //assert(0 == hitsModuleStart[0]);

    if (i < 11) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      out << "LayerStart " << i << " " << cpeParams->layerGeometry().layerStart[i] << ": " << hitsLayerStart[i]
             << sycl::endl;
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           sycl::queue queue) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), queue);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      queue.submit([&](sycl::handler& cgh) {
        sycl::stream out(64 * 1024, 80, cgh);
        sycl::accessor<pixelCPEforGPU::ClusParams, 0, sycl::access_mode::read_write, sycl::target::local> clusParams_acc(
            cgh);

        auto bs_d_data = bs_d.data();
        auto digis_d_view = digis_d.view();
        auto digis_d_nDigis = digis_d.nDigis();
        auto clusters_d_view = clusters_d.view();
        auto hits_d_view = hits_d.view();

        cgh.parallel_for(
            sycl::nd_range(sycl::range(1, 1, blocks * threadsPerBlock), sycl::range(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item) {
              gpuPixelRecHits::getHits(cpeParams,
                                       bs_d_data,
                                       digis_d_view,
                                       digis_d_nDigis,
                                       clusters_d_view,
                                       hits_d_view,
                                       item,
                                       out,
                                       clusParams_acc.get_pointer());
            });
      });
#ifdef GPU_DEBUG
    queue.wait_and_throw();
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::stream out(64 * 1024, 80, cgh);

        auto clusters_d_clusModuleStart = clusters_d.clusModuleStart();
        auto hits_d_hitsLayerStart = hits_d.hitsLayerStart();

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 32), sycl::range(1, 1, 32)), [=](sycl::nd_item<3> item) {
          setHitsLayerStart(clusters_d_clusModuleStart, cpeParams, hits_d_hitsLayerStart, item, out);
        });
      });
    }

    if (nHits) {
      cms::sycltools::fillManyFromVector(
          hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, queue);
    }

#ifdef GPU_DEBUG
    queue.wait_and_throw();
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
