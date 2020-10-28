#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"
#include "SYCLCore/copyAsync.h"

namespace testTrackingRecHit2D {

  void fill(TrackingRecHit2DSOAView* phits, sycl::nd_item<3> item) {
    //assert(phits);
    auto& hits = *phits;
    //assert(hits.nHits() == 200);

    int i = item.get_local_id(2);
    if (i > 200)
      return;
  }

  void verify(TrackingRecHit2DSOAView const* phits, sycl::nd_item<3> item) {
    //assert(phits);
    auto const& hits = *phits;
    //assert(hits.nHits() == 200);

    int i = item.get_local_id(2);
    if (i > 200)
      return;
  }

  void runKernels(sycl::queue queue, TrackingRecHit2DSOAView* hits) {
    assert(hits);
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
                       [=](sycl::nd_item<3> item) { fill(hits, item); });
    });
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
                       [=](sycl::nd_item<3> item) { verify(hits, item); });
    });
    queue.wait_and_throw();
  }

}  // namespace testTrackingRecHit2D


int main() {
  sycl::queue queue = dpct::get_default_queue();

  auto nHits = 200;
  TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, queue);

  testTrackingRecHit2D::runKernels(queue, tkhit.view());

  return 0;
}
