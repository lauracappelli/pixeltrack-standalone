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
                         sycl::nd_item<3> item_ct1) {
    auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

    assert(0 == hitsModuleStart[0]);

    if (i < 11) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           sycl::queue stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      /*
      DPCT1049:66: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
      */
      stream.submit([&](sycl::handler& cgh) {
        sycl::accessor<ClusParams, 0, sycl::access::mode::read_write, sycl::access::target::local> clusParams_acc_ct1(
            cgh);

        auto bs_d_data_ct1 = bs_d.data();
        auto digis_d_view_ct2 = digis_d.view();
        auto digis_d_nDigis_ct3 = digis_d.nDigis();
        auto clusters_d_view_ct4 = clusters_d.view();
        auto hits_d_view_ct5 = hits_d.view();

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, blocks) * sycl::range(1, 1, threadsPerBlock),
                                        sycl::range(1, 1, threadsPerBlock)),
                         [=](sycl::nd_item<3> item_ct1) {
                           getHits(cpeParams,
                                   bs_d_data_ct1,
                                   digis_d_view_ct2,
                                   digis_d_nDigis_ct3,
                                   clusters_d_view_ct4,
                                   hits_d_view_ct5,
                                   item_ct1,
                                   clusParams_acc_ct1.get_pointer());
                         });
      });
#ifdef GPU_DEBUG
    stream.wait_and_throw();
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      stream.submit([&](sycl::handler& cgh) {
        auto clusters_d_clusModuleStart_ct0 = clusters_d.clusModuleStart();
        auto hits_d_hitsLayerStart_ct2 = hits_d.hitsLayerStart();

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 32), sycl::range(1, 1, 32)), [=](sycl::nd_item<3> item_ct1) {
          setHitsLayerStart(clusters_d_clusModuleStart_ct0, cpeParams, hits_d_hitsLayerStart_ct2, item_ct1);
        });
      });
    }

    if (nHits) {
      auto hws = cms::sycltools::make_device_unique<uint8_t[]>(TrackingRecHit2DSOAView::Hist::wsSize(), stream);
      cms::sycltools::fillManyFromVector(
          hits_d.phiBinner(), hws.get(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
    }

#ifdef GPU_DEBUG
    stream.wait_and_throw();
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
