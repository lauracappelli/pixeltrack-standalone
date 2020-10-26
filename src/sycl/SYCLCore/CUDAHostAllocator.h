#ifndef HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
#define HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>
#include <new>

class cuda_bad_alloc : public std::bad_alloc {
public:
  cuda_bad_alloc(int error) noexcept : error_(error) {}

  /*
  DPCT1009:20: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
  */
  const char* what() const noexcept override {
    return "cudaGetErrorString not supported" /*cudaGetErrorString(error_)*/;
  }

private:
  int error_;
};

/*
DPCT1048:9: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
*/
template <typename T, unsigned int FLAGS = 0>
class CUDAHostAllocator {
public:
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = CUDAHostAllocator<U, FLAGS>;
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
