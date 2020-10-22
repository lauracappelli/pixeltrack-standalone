#ifndef HeterogeneousCore_CUDAUtilities_interface_host_noncached_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_host_noncached_unique_ptr_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>

#include "SYCLCore/cudaCheck.h"

namespace cms {
  namespace sycltools {
    namespace host {
      namespace noncached {
        namespace impl {
          // Additional layer of types to distinguish from host::unique_ptr
          class HostDeleter {
          public:
            /*
            DPCT1003:62: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
            */
            void operator()(void *ptr) { cudaCheck((sycl::free(ptr, dpct::get_default_queue()), 0)); }
          };
        }  // namespace impl

        template <typename T>
        using unique_ptr = std::unique_ptr<T, impl::HostDeleter>;

        namespace impl {
          template <typename T>
          struct make_host_unique_selector {
            using non_array = cms::sycltools::host::noncached::unique_ptr<T>;
          };
          template <typename T>
          struct make_host_unique_selector<T[]> {
            using unbounded_array = cms::sycltools::host::noncached::unique_ptr<T[]>;
          };
          template <typename T, size_t N>
          struct make_host_unique_selector<T[N]> {
            struct bounded_array {};
          };
        }  // namespace impl
      }    // namespace noncached
    }      // namespace host

    /**
   * The difference wrt. make_host_unique is that these
   * do not cache, so they should not be called per-event.
   */
    template <typename T>
    typename host::noncached::impl::make_host_unique_selector<T>::non_array make_host_noncached_unique(
        /*
        DPCT1048:60: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
        */
        unsigned int flags = 0) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem;
      /*
      DPCT1003:63: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((mem = (void *)sycl::malloc_host(sizeof(T), dpct::get_default_queue()), 0));
      return typename host::noncached::impl::make_host_unique_selector<T>::non_array(reinterpret_cast<T *>(mem));
    }

    template <typename T>
    typename host::noncached::impl::make_host_unique_selector<T>::unbounded_array make_host_noncached_unique(
        /*
        DPCT1048:61: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
        */
        size_t n,
        unsigned int flags = 0) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem;
      /*
      DPCT1003:64: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((mem = (void *)sycl::malloc_host(n * sizeof(element_type), dpct::get_default_queue()), 0));
      return typename host::noncached::impl::make_host_unique_selector<T>::unbounded_array(
          reinterpret_cast<element_type *>(mem));
    }

    template <typename T, typename... Args>
    typename host::noncached::impl::make_host_unique_selector<T>::bounded_array make_host_noncached_unique(Args &&...) =
        delete;
  }  // namespace sycltools
}  // namespace cms

#endif
