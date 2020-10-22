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
  __attribute__((returns_nonnull)) try {
    void* ptr = nullptr;
    /*
    DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    int status = (ptr = (void*)sycl::malloc_host(n * sizeof(T), dpct::get_default_queue()), 0);
    /*
    DPCT1000:22: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status != 0) {
      /*
      DPCT1001:21: The statement could not be removed.
      */
      throw cuda_bad_alloc(status);
    }
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void deallocate(T* p, std::size_t n) const try {
    /*
    DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    int status = (sycl::free(p, dpct::get_default_queue()), 0);
    /*
    DPCT1000:25: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status != 0) {
      /*
      DPCT1001:24: The statement could not be removed.
      */
      throw cuda_bad_alloc(status);
    }
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
};

#endif  // HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
