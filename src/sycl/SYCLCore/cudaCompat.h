#ifndef HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
#define HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h

/*
 * Everything you need to run cuda code in plain sequential c++ code
 */

#ifndef CL_SYCL_LANGUAGE_VERSION

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace cudaCompat {

#ifndef __DPCT_HPP__
  struct dim3 {
    uint32_t x, y, z;
  };
#endif
  const sycl::range<3> threadIdx = {0, 0, 0};
  const sycl::range<3> blockDim = {1, 1, 1};

  extern thread_local sycl::range blockIdx;
  extern thread_local sycl::range gridDim;

  template <typename T1, typename T2>
  T1 atomicInc(T1* a, T2 b) {
    auto ret = *a;
    if ((*a) < T1(b))
      (*a)++;
    return ret;
  }

  template <typename T1, typename T2>
  T1 atomicAdd(T1* a, T2 b) {
    auto ret = *a;
    (*a) += b;
    return ret;
  }

  template <typename T1, typename T2>
  T1 atomicSub(T1* a, T2 b) {
    auto ret = *a;
    (*a) -= b;
    return ret;
  }

  template <typename T1, typename T2>
  T1 atomicMin(T1* a, T2 b) {
    auto ret = *a;
    *a = std::min(*a, T1(b));
    return ret;
  }
  template <typename T1, typename T2>
  T1 atomicMax(T1* a, T2 b) {
    auto ret = *a;
    *a = std::max(*a, T1(b));
    return ret;
  }

  inline void __syncthreads() {}
  inline void __threadfence() {}
  inline bool __syncthreads_or(bool x) { return x; }
  inline bool __syncthreads_and(bool x) { return x; }
  template <typename T>
  inline T __ldg(T const* x) {
    return *x;
  }

  inline void resetGrid() {
    blockIdx = sycl::range(0, 0, 0);
    gridDim = sycl::range(1, 1, 1);
  }

}  // namespace cudaCompat

// some  not needed as done by cuda runtime...
#ifndef __DPCT_HPP__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __forceinline__
#endif

// make sure function are inlined to avoid multiple definition
#ifndef DPCPP_COMPATIBILITY_TEMP
#undef __global__
#define __global__ inline __attribute__((always_inline))
#undef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#ifndef DPCPP_COMPATIBILITY_TEMP
using namespace cudaCompat;
#endif

#endif

#endif  // HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
