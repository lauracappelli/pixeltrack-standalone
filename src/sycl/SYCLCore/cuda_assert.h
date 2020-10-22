// The omission of #include guards is on purpose: it does make sense to #include
// this file multiple times, setting a different value of GPU_DEBUG beforehand.

#ifdef DPCPP_COMPATIBILITY_TEMP
#ifndef GPU_DEBUG
// disable asserts
#ifndef NDEBUG
#define NDEBUG
#endif
#else
// enable asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#endif
#endif  // __CUDA_ARCH__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
