#ifndef HeterogeneousCoreCUDAUtilities_radixSort_H
#define HeterogeneousCoreCUDAUtilities_radixSort_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

#include "SYCLCore/cuda_assert.h"

template <typename T>
inline void dummyReorder(T const *a, uint16_t *ind, uint16_t *ind2, uint32_t size) {}

template <typename T>
inline void reorderSigned(
    T const *a, uint16_t *ind, uint16_t *ind2, uint32_t size, sycl::nd_item<3> item_ct1, uint32_t *firstNeg) {
  //move negative first...

  int32_t first = item_ct1.get_local_id(2);

  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  item_ct1.barrier();

  // find first negative
  for (auto i = first; i < size - 1; i += item_ct1.get_local_range().get(2)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  item_ct1.barrier();

  auto ii = first;
  for (auto i = *firstNeg + item_ct1.get_local_id(2); i < size; i += item_ct1.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();
  ii = size - *firstNeg + item_ct1.get_local_id(2);
  assert(ii >= 0);
  for (auto i = first; i < *firstNeg; i += item_ct1.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    ind[i] = ind2[i];
}

template <typename T>
inline void reorderFloat(
    T const *a, uint16_t *ind, uint16_t *ind2, uint32_t size, sycl::nd_item<3> item_ct1, uint32_t *firstNeg) {
  //move negative first...

  int32_t first = item_ct1.get_local_id(2);

  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  item_ct1.barrier();

  // find first negative
  for (auto i = first; i < size - 1; i += item_ct1.get_local_range().get(2)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  item_ct1.barrier();

  int ii = size - *firstNeg - item_ct1.get_local_id(2) - 1;
  for (auto i = *firstNeg + item_ct1.get_local_id(2); i < size; i += item_ct1.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii -= item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();
  ii = size - *firstNeg + item_ct1.get_local_id(2);
  assert(ii >= 0);
  for (auto i = first; i < *firstNeg; i += item_ct1.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    ind[i] = ind2[i];
}

template <typename T,  // shall be interger
          int NS,      // number of significant bytes to use in sorting
          typename RF>
void __dpct_inline__ radixSortImpl(T const *__restrict__ a,
                                   uint16_t *ind,
                                   uint16_t *ind2,
                                   uint32_t size,
                                   RF reorder,
                                   sycl::nd_item<3> item_ct1,
                                   int32_t *c,
                                   int32_t *ct,
                                   int32_t *cu,
                                   int *ibs,
                                   int *p) {
  constexpr int d = 8, w = 8 * sizeof(T);
  constexpr int sb = 1 << d;
  constexpr int ps = int(sizeof(T)) - NS;

  assert(size > 0);
  assert(blockDim.x >= sb);

  // bool debug = false; // threadIdx.x==0 && blockIdx.x==5;

  *p = ps;

  auto j = ind;
  auto k = ind2;

  int32_t first = item_ct1.get_local_id(2);
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    j[i] = i;
  item_ct1.barrier();

  while ((item_ct1.barrier(), sycl::intel::all_of(item_ct1.get_group(), p < w / d))) {
    if (item_ct1.get_local_id(2) < sb)
      c[item_ct1.get_local_id(2)] = 0;
    item_ct1.barrier();

    // fill bins
    for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {
      auto bin = (a[j[i]] >> d * *p) & (sb - 1);
      sycl::atomic<int32_t, sycl::access::address_space::local_space>(sycl::local_ptr<int32_t>(&c[bin])).fetch_add(1);
    }
    item_ct1.barrier();

    // prefix scan "optimized"???...
    if (item_ct1.get_local_id(2) < sb) {
      auto x = c[item_ct1.get_local_id(2)];
      auto laneId = item_ct1.get_local_id(2) & 0x1f;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        /*
        DPCT1023:207: The DPC++ sub-group does not support mask options for shuffle_up.
        */
        auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
        if (laneId >= offset)
          x += y;
      }
      ct[item_ct1.get_local_id(2)] = x;
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < sb) {
      auto ss = (item_ct1.get_local_id(2) / 32) * 32 - 1;
      c[item_ct1.get_local_id(2)] = ct[item_ct1.get_local_id(2)];
      for (int i = ss; i > 0; i -= 32)
        c[item_ct1.get_local_id(2)] += ct[i];
    }
    /* 
    //prefix scan for the nulls  (for documentation)
    if (threadIdx.x==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

    // broadcast
    *ibs = size - 1;
    item_ct1.barrier();
    while ((item_ct1.barrier(), sycl::intel::all_of(item_ct1.get_group(), ibs > 0))) {
      int i = *ibs - item_ct1.get_local_id(2);
      if (item_ct1.get_local_id(2) < sb) {
        cu[item_ct1.get_local_id(2)] = -1;
        ct[item_ct1.get_local_id(2)] = -1;
      }
      item_ct1.barrier();
      int32_t bin = -1;
      if (item_ct1.get_local_id(2) < sb) {
        if (i >= 0) {
          bin = (a[j[i]] >> d * *p) & (sb - 1);
          ct[item_ct1.get_local_id(2)] = bin;
          sycl::atomic<int32_t, sycl::access::address_space::local_space>(sycl::local_ptr<int32_t>(&cu[bin]))
              .fetch_max(int(i));
        }
      }
      item_ct1.barrier();
      if (item_ct1.get_local_id(2) < sb) {
        if (i >= 0 && i == cu[bin])  // ensure to keep them in order
          for (int ii = item_ct1.get_local_id(2); ii < sb; ++ii)
            if (ct[ii] == bin) {
              auto oi = ii - item_ct1.get_local_id(2);
              // assert(i>=oi);if(i>=oi)
              k[--c[bin]] = j[i - oi];
            }
      }
      item_ct1.barrier();
      if (bin >= 0)
        assert(c[bin] >= 0);
      if (item_ct1.get_local_id(2) == 0)
        *ibs -= sb;
      item_ct1.barrier();
    }

    /*
    // broadcast for the nulls  (for documentation)
    if (threadIdx.x==0)
    for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      auto ik = atomicSub(&c[bin],1);
      k[ik-1] = j[i];
    }
    */

    item_ct1.barrier();
    assert(c[0] == 0);

    // swap (local, ok)
    auto t = j;
    j = k;
    k = t;

    if (item_ct1.get_local_id(2) == 0)
      ++(*p);
    item_ct1.barrier();
  }

  if ((w != 8) && (0 == (NS & 1)))
    assert(j == ind);  // w/d is even so ind is correct

  if (j != ind)  // odd...
    for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
      ind[i] = ind2[i];

  item_ct1.barrier();

  // now move negative first... (if signed)
  reorder(a, ind, ind2, size);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_unsigned<T>::value, T>::type * = nullptr>
void __dpct_inline__ radixSort(T const *a,
                               uint16_t *ind,
                               uint16_t *ind2,
                               uint32_t size,
                               sycl::nd_item<3> item_ct1,
                               int32_t *c,
                               int32_t *ct,
                               int32_t *cu,
                               int *ibs,
                               int *p) {
  radixSortImpl<T, NS>(a, ind, ind2, size, dummyReorder<T>, item_ct1, c, ct, cu, ibs, p);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type * = nullptr>
void __dpct_inline__ radixSort(T const *a,
                               uint16_t *ind,
                               uint16_t *ind2,
                               uint32_t size,
                               sycl::nd_item<3> item_ct1,
                               int32_t *c,
                               int32_t *ct,
                               int32_t *cu,
                               int *ibs,
                               int *p) {
  radixSortImpl<T, NS>(a, ind, ind2, size, reorderSigned<T>, item_ct1, c, ct, cu, ibs, p);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_floating_point<T>::value, T>::type * = nullptr>
void __dpct_inline__ radixSort(T const *a,
                               uint16_t *ind,
                               uint16_t *ind2,
                               uint32_t size,
                               sycl::nd_item<3> item_ct1,
                               int32_t *c,
                               int32_t *ct,
                               int32_t *cu,
                               int *ibs,
                               int *p) {
  using I = int;
  radixSortImpl<I, NS>((I const *)(a), ind, ind2, size, reorderFloat<I>, item_ct1, c, ct, cu, ibs, p);
}

template <typename T, int NS = sizeof(T)>
void __dpct_inline__ radixSortMulti(T const *v,
                                    uint16_t *index,
                                    uint32_t const *offsets,
                                    uint16_t *workspace,
                                    sycl::nd_item<3> item_ct1,
                                    uint8_t *dpct_local,
                                    int32_t *c,
                                    int32_t *ct,
                                    int32_t *cu,
                                    int *ibs,
                                    int *p) {
  auto ws = (uint16_t *)dpct_local;

  auto a = v + offsets[item_ct1.get_group(2)];
  auto ind = index + offsets[item_ct1.get_group(2)];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[item_ct1.get_group(2)];
  auto size = offsets[item_ct1.get_group(2) + 1] - offsets[item_ct1.get_group(2)];
  assert(offsets[blockIdx.x + 1] >= offsets[blockIdx.x]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size, item_ct1, c, ct, cu, ibs, p);
}

template <typename T, int NS = sizeof(T)>
void radixSortMultiWrapper(T const *v,
                           uint16_t *index,
                           uint32_t const *offsets,
                           uint16_t *workspace,
                           sycl::nd_item<3> item_ct1,
                           uint8_t *dpct_local,
                           int32_t *c,
                           int32_t *ct,
                           int32_t *cu,
                           int *ibs,
                           int *p) {
  radixSortMulti<T, NS>(v, index, offsets, workspace, item_ct1, dpct_local, c, ct, cu, ibs, p);
}

template <typename T, int NS = sizeof(T)>
void
// __launch_bounds__(256, 4)
radixSortMultiWrapper2(T const *v,
                       uint16_t *index,
                       uint32_t const *offsets,
                       uint16_t *workspace,
                       sycl::nd_item<3> item_ct1,
                       uint8_t *dpct_local,
                       int32_t *c,
                       int32_t *ct,
                       int32_t *cu,
                       int *ibs,
                       int *p) {
  radixSortMulti<T, NS>(v, index, offsets, workspace, item_ct1, dpct_local, c, ct, cu, ibs, p);
}

#endif  // HeterogeneousCoreCUDAUtilities_radixSort_H
