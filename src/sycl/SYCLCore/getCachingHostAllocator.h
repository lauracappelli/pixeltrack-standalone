#ifndef HeterogeneousCore_SYCLCore_src_getCachingHostAllocator
#define HeterogeneousCore_SYCLCore_src_getCachingHostAllocator

#include "SYCLCore/cudaCheck.h"
#include "CachingHostAllocator.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda::allocator {
  inline notcub::CachingHostAllocator& getCachingHostAllocator() {
    // the public interface is thread safe
    static notcub::CachingHostAllocator allocator{binGrowth,
                                                  minBin,
                                                  maxBin,
                                                  minCachedBytes(),
                                                  false,  // do not skip cleanup
                                                  debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
