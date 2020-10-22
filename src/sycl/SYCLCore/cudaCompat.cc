#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLCore/cudaCompat.h"

namespace sycltoolsCompat {
  thread_local sycl::range blockIdx(1, 1, 1);
  thread_local sycl::range gridDim(1, 1, 1);
}  // namespace sycltoolsCompat

namespace {
  struct InitGrid {
    InitGrid() { cudaCompat::resetGrid(); }
  };

  const InitGrid initGrid;

}  // namespace
