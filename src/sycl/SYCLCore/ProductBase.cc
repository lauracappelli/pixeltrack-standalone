#include <CL/sycl.hpp>

#include "SYCLCore/ProductBase.h"
#include "SYCLCore/eventWorkHasCompleted.h"

namespace cms::sycltools {
  bool ProductBase::isAvailable() const { return eventWorkHasCompleted(event_); }

  ProductBase::~ProductBase() {
    // Make sure that the production of the product in the GPU is
    // complete before destructing the product. This is to make sure
    // that the EDM stream does not move to the next event before all
    // asynchronous processing of the current is complete.

    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.
    if (event_) {
      event_->wait_and_throw();
    }
  }
}  // namespace cms::sycltools
