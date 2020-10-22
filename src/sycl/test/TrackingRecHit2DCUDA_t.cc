#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"
#include "SYCLCore/copyAsync.h"

namespace testTrackingRecHit2D {

  void fill(TrackingRecHit2DSOAView* phits, sycl::nd_item<3> item_ct1) {
    assert(phits);
    auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = item_ct1.get_local_id(2);
    if (i > 200)
      return;
  }

  void verify(TrackingRecHit2DSOAView const* phits, sycl::nd_item<3> item_ct1) {
    assert(phits);
    auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = item_ct1.get_local_id(2);
    if (i > 200)
      return;
  }

  void runKernels(TrackingRecHit2DSOAView* hits) {
    assert(hits);
    /*
    DPCT1049:56: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
                       [=](sycl::nd_item<3> item_ct1) { fill(hits, item_ct1); });
    });
    /*
    DPCT1049:57: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
                       [=](sycl::nd_item<3> item_ct1) { verify(hits, item_ct1); });
    });
  }

}  // namespace testTrackingRecHit2D

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView* hits);

}

int main() {
  sycl::queue* stream;
  /*
  DPCT1025:58: The SYCL queue is created ignoring the flag/priority options.
  */
  stream = dpct::get_current_device().create_queue(true);

  // inner scope to deallocate memory before destroying the stream
  {
    auto nHits = 200;
    TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.view());
  }

  dpct::get_current_device().destroy_queue(stream);

  return 0;
}
