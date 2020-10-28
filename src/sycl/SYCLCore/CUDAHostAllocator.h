#ifndef HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
#define HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h

#include <memory>
#include <new>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

template <typename T>
class CUDAHostAllocator {
public:
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = CUDAHostAllocator<U>;
  };

  T* allocate(std::size_t n) const __attribute__((warn_unused_result)) __attribute__((malloc))
  __attribute__((returns_nonnull)) {
    void* ptr = (void*)sycl::malloc_host(n * sizeof(T), dpct::get_default_queue());
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t n) const {
    sycl::free(p, dpct::get_default_queue());
  }
};

#endif  // HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
