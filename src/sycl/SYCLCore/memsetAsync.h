#ifndef HeterogeneousCore_CUDAUtilities_memsetAsync_h
#define HeterogeneousCore_CUDAUtilities_memsetAsync_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"

#include <type_traits>

namespace cms {
  namespace sycltools {
    template <typename T>
    inline void memsetAsync(device::unique_ptr<T>& ptr, T value, sycl::queue* stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      /*
      DPCT1003:70: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((stream->memset(ptr.get(), value, sizeof(T)), 0));
    }

    /**
   * The type of `value` is `int` because of `cudaMemsetAsync()` takes
   * it as an `int`. Note that `cudaMemsetAsync()` sets the value of
   * each **byte** to `value`. This may lead to unexpected results if
   * `sizeof(T) > 1` and `value != 0`.
   */
    template <typename T>
    inline void memsetAsync(device::unique_ptr<T[]>& ptr, int value, size_t nelements, sycl::queue* stream) {
      /*
      DPCT1003:71: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((stream->memset(ptr.get(), value, nelements * sizeof(T)), 0));
    }
  }  // namespace sycltools
}  // namespace cms

#endif
