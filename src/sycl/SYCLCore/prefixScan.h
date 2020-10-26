#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cassert>
#include <cstdint>

#include <CL/sycl.hpp>

template <typename T>
SYCL_EXTERNAL ALWAYS_INLINE void warpPrefixScan(
    T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask, sycl::nd_item<3> item_ct1) {
#ifdef DPCT_NEED_WORK
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = item_ct1.get_local_id(2) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    /*
    DPCT1023:57: The DPC++ sub-group does not support mask options for shuffle_up.
    */
    auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
#endif  // DPCT_NEED_WORK
}

//same as above may remove
template <typename T>
SYCL_EXTERNAL ALWAYS_INLINE void warpPrefixScan(T* c, uint32_t i, uint32_t mask, sycl::nd_item<3> item_ct1) {
#ifdef DPCT_NEED_WORK
  auto x = c[i];
  auto laneId = item_ct1.get_local_id(2) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    /*
    DPCT1023:58: The DPC++ sub-group does not support mask options for shuffle_up.
    */
    auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
#endif  // DPCT_NEED_WORK
}

// limited to 32*32 elements....
template <typename T>
SYCL_EXTERNAL ALWAYS_INLINE void blockPrefixScan(
    T const* __restrict__ ci, T* __restrict__ co, uint32_t size, T* ws, sycl::nd_item<3> item_ct1) {
#ifdef DPCT_NEED_WORK
  assert(ws);
  assert(size <= 1024);
  assert(0 == item_ct1.get_local_range().get(2) % 32);
  auto first = item_ct1.get_local_id(2);
  /*
  DPCT1004:19: Could not generate replacement.
  */
  auto mask = __ballot_sync(0xffffffff, first < size);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {
    warpPrefixScan(ci, co, i, item_ct1, mask);
    auto laneId = item_ct1.get_local_id(2) & 0x1f;
    auto warpId = i / 32;
    assert(warpId < 32);
    if (31 == laneId)
      ws[warpId] = co[i];
    /*
    DPCT1004:20: Could not generate replacement.
    */
    mask = __ballot_sync(mask, i + item_ct1.get_local_range(2) < size);
  }
  item_ct1.barrier();
  if (size <= 32)
    return;
  if (item_ct1.get_local_id(2) < 32)
    warpPrefixScan(ws, item_ct1.get_local_id(2), 0xffffffff, item_ct1);
  item_ct1.barrier();
  for (auto i = first + 32; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / 32;
    co[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#endif  // DPCT_NEED_WORK
}

// same as above, may remove
// limited to 32*32 elements....
template <typename T>
SYCL_EXTERNAL ALWAYS_INLINE void blockPrefixScan(T* c, uint32_t size, T* ws, sycl::nd_item<3> item_ct1) {
#ifdef DPCT_NEED_WORK
  assert(ws);
  assert(size <= 1024);
  assert(0 == item_ct1.get_local_range().get(2) % 32);
  auto first = item_ct1.get_local_id(2);
  /*
  DPCT1004:21: Could not generate replacement.
  */
  auto mask = __ballot_sync(0xffffffff, first < size);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {
    warpPrefixScan(c, i, mask, item_ct1);
    auto laneId = item_ct1.get_local_id(2) & 0x1f;
    auto warpId = i / 32;
    assert(warpId < 32);
    if (31 == laneId)
      ws[warpId] = c[i];
    /*
    DPCT1004:22: Could not generate replacement.
    */
    mask = __ballot_sync(mask, i + item_ct1.get_local_range(2) < size);
  }
  item_ct1.barrier();
  if (size <= 32)
    return;
  if (item_ct1.get_local_id(2) < 32)
    warpPrefixScan(ws, item_ct1.get_local_id(2), 0xffffffff, item_ct1);
  item_ct1.barrier();
  for (auto i = first + 32; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / 32;
    c[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#endif  // DPCT_NEED_WORK
}

// limited to 1024*1024 elements....
template <typename T>
void multiBlockPrefixScan(T const* __restrict__ ci,
                          T* __restrict__ co,
                          int32_t size,
                          int32_t* pc,
                          sycl::nd_item<3> item_ct1,
                          T* ws,
                          bool* isLastBlockDone,
                          T* psum) {
  // first each block does a scan of size 1024; (better be enough blocks....)
  assert(size > 0 && 1024 * item_ct1.get_group_range(2) >= (unsigned int) size);
  int off = 1024 * item_ct1.get_group(2);
  if (size - off > 0)
    blockPrefixScan(ci + off, co + off, sycl::min(1024, size - off), item_ct1, ws);

  // count blocks that finished

  if (0 == item_ct1.get_local_id(2)) {
    auto value = sycl::atomic<int32_t>(sycl::global_ptr<int32_t>(pc)).fetch_add(1);  // block counter
    *isLastBlockDone = (value == (int(item_ct1.get_group_range(2)) - 1));
  }

  item_ct1.barrier();

  if (!(*isLastBlockDone))
    return;

  // good each block has done its work and now we are left in last block

  // let's get the partial sums from each block

  for (int i = item_ct1.get_local_id(2), ni = item_ct1.get_group_range(2); i < ni;
       i += item_ct1.get_local_range().get(2)) {
    auto j = 1024 * i + 1023;
    psum[i] = (j < size) ? co[j] : T(0);
  }
  item_ct1.barrier();
  blockPrefixScan(psum, psum, item_ct1.get_group_range(2), item_ct1, ws);

  // now it would have been handy to have the other blocks around...
  int first = item_ct1.get_local_id(2);                                           // + blockDim.x * blockIdx.x
  for (int i = first + 1024; i < size; i += item_ct1.get_local_range().get(2)) {  //  *gridDim.x) {
    auto k = i / 1024;                                                            // block
    co[i] += psum[k - 1];
  }
}

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
