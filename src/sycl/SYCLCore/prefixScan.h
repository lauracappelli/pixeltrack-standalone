#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cassert>
#include <cstdint>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {

    template <typename T>
    inline __attribute__((always_inline)) void warpPrefixScan(
        sycl::nd_item<3> item, T const* __restrict__ ci, T* __restrict__ co, uint32_t i, unsigned int subgroupSize) {
      // ci and co may be the same
      auto x = ci[i];
      auto laneId = item.get_local_id(2) % subgroupSize;
#pragma unroll
      for (unsigned int offset = 1; offset < subgroupSize; offset <<= 1) {
        auto y = item.get_sub_group().shuffle_up(x, offset);
        if (laneId >= offset)
          x += y;
      }
      co[i] = x;
    }

    //same as above may remove
    template <typename T>
    inline __attribute__((always_inline)) void warpPrefixScan(sycl::nd_item<3> item,
                                                              T* c,
                                                              uint32_t i,
                                                              unsigned int subgroupSize) {
      auto x = c[i];
      auto laneId = item.get_local_id(2) % subgroupSize;
#pragma unroll
      for (unsigned int offset = 1; offset < subgroupSize; offset <<= 1) {
        auto y = item.get_sub_group().shuffle_up(x, offset);
        if (laneId >= offset)
          x += y;
      }
      c[i] = x;
    }

    // limited to 32*32 elements....
    template <typename VT, typename T>
    inline __attribute__((always_inline)) void blockPrefixScan(sycl::nd_item<3> item,
                                                               VT const* __restrict__ ci,
                                                               VT* __restrict__ co,
                                                               uint32_t size,
                                                               T* ws,
                                                               sycl::stream out,
                                                               unsigned int subgroupSize) {
      //assert(ws);
      if (!ws) {
        out << "failed (blockPrefixScan): != ws " << sycl::endl;
        return;
      }
      //assert(size <= 1024);
      if (size > 1024) {
        out << "failed (blockPrefixScan): size > 1024 " << sycl::endl;
        return;
      }
      //assert(0 == blockDim.x % 32);
      if (0 != item.get_local_range(2) % subgroupSize) {
        out << "failed (blockPrefixScan): 0 != item.get_local_range(2) % subgroupSize " << sycl::endl;
        return;
      }

      auto first = item.get_local_id(2);

      for (auto i = first; i < size; i += item.get_local_range().get(2)) {
        warpPrefixScan(item, ci, co, i, subgroupSize);
        auto laneId = item.get_local_id(2) % subgroupSize;
        auto warpId = i / subgroupSize;
        //assert(warpId < 32);
        if (warpId >= subgroupSize) {
          out << "failed (blockPrefixScan): warpId >= subgroupSize " << sycl::endl;
          return;
        }
        if (subgroupSize - 1 == laneId)
          ws[warpId] = co[i];
      }
      item.barrier();
      if (size <= subgroupSize)
        return;
      if (item.get_local_id(2) < subgroupSize)
        warpPrefixScan(item, ws, item.get_local_id(2), subgroupSize);
      item.barrier();
      for (auto i = first + subgroupSize; i < size; i += item.get_local_range().get(2)) {
        auto warpId = i / subgroupSize;
        co[i] += ws[warpId - 1];
      }
      item.barrier();
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    inline __attribute__((always_inline)) void blockPrefixScan(
        sycl::nd_item<3> item, T* c, uint32_t size, T* ws, sycl::stream out, unsigned int subgroupSize) {
      //assert(ws);
      if (!ws) {
        out << "failed (blockPrefixScan): != ws " << sycl::endl;
        return;
      }
      //assert(size <= 1024);
      if (size > 1024) {
        out << "failed (blockPrefixScan): size > 1024 " << sycl::endl;
        return;
      }
      //assert(0 == blockDim.x % 32);
      if (0 != item.get_local_range().get(2) % subgroupSize) {
        out << "failed (blockPrefixScan): 0 != item.get_local_range(2) % subgroupSize " << sycl::endl;
        return;
      }

      auto first = item.get_local_id(2);

      for (auto i = first; i < size; i += item.get_local_range().get(2)) {
        warpPrefixScan(item, c, i, subgroupSize);
        auto laneId = item.get_local_id(2) % subgroupSize;
        auto warpId = i / subgroupSize;
        //assert(warpId < 32);
        if (warpId >= subgroupSize) {
          out << "failed (blockPrefixScan): warpId >= subgroupSize " << sycl::endl;
          return;
        }
        if ((subgroupSize - 1) == laneId)
          ws[warpId] = c[i];
      }
      item.barrier();
      if (size <= subgroupSize)
        return;
      if (item.get_local_id(2) < subgroupSize)
        warpPrefixScan(item, ws, item.get_local_id(2), subgroupSize);
      item.barrier();
      for (auto i = first + subgroupSize; i < size; i += item.get_local_range().get(2)) {
        auto warpId = i / subgroupSize;
        c[i] += ws[warpId - 1];
      }
      item.barrier();
    }

    template <typename T>
    void multiBlockPrefixScan(sycl::nd_item<3> item,
                              T* const ci,
                              T* co,
                              uint32_t size,
                              int32_t* pc,
                              T* ws,
                              bool* isLastBlockDone,
                              T* psum,
                              sycl::stream out,
                              unsigned int subgroupSize) {
      //assert(1024 * gridDim.x >= size);
      if (item.get_local_range().get(2) * item.get_group_range().get(2) < size) {
        out << "failed (multiBlockPrefixScan): item.get_local_range().get(2) * item.get_group_range.get(2) < size "
            << sycl::endl;
        return;
      }
      int off = item.get_local_range().get(2) * item.get_group(2);
      if (size - off > 0)
        blockPrefixScan(
            item, ci + off, co + off, sycl::min<uint32_t>(item.get_local_range(2), size - off), ws, out, subgroupSize);

      // count blocks that finished

      if (0 == item.get_local_id(2)) {
        auto value = sycl::atomic<int32_t>(sycl::global_ptr<int32_t>(pc)).fetch_add(1);  // block counter
        *isLastBlockDone = (value == (int(item.get_group_range(2)) - 1));
      }
      item.barrier();

      if (!(*isLastBlockDone))
        return;

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      for (unsigned int i = item.get_local_id(2), ni = item.get_group_range(2); i < ni;
           i += item.get_local_range().get(2)) {
        auto j = item.get_local_range().get(2) * i + item.get_local_range().get(2) - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      item.barrier();
      blockPrefixScan(item, psum, psum, item.get_group_range(2), ws, out, subgroupSize);

      // now it would have been handy to have the other blocks around...
      for (unsigned int i = item.get_local_id(2) + item.get_local_range().get(2), k = 0; i < size;
           i += item.get_local_range().get(2), ++k) {
        co[i] += psum[k];
      }
    }

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
