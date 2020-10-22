#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <CL/sycl.hpp>

#ifdef CL_SYCL_LANGUAGE_VERSION
#include <cub/cub.cuh>
#endif

#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/cudastdAlgorithm.h"
#include "SYCLCore/prefixScan.h"

namespace cms {
  namespace sycltools {

    template <typename Histo, typename T>
    void countFromVector(Histo *__restrict__ h,
                         uint32_t nh,
                         T const *__restrict__ v,
                         uint32_t const *__restrict__ offsets,
                         sycl::nd_item<3> item_ct1) {
      int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
      for (int i = first, nt = offsets[nh]; i < nt;
           i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).count(v[i], ih);
      }
    }

    template <typename Histo, typename T>
    void fillFromVector(Histo *__restrict__ h,
                        uint32_t nh,
                        T const *__restrict__ v,
                        uint32_t const *__restrict__ offsets,
                        sycl::nd_item<3> item_ct1) {
      int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
      for (int i = first, nt = offsets[nh]; i < nt;
           i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo>
    inline void launchZero(Histo *__restrict__ h,
                           sycl::queue *stream
#ifndef CL_SYCL_LANGUAGE_VERSION
                           = cudaStreamDefault
#endif
    ) {
      uint32_t *off = (uint32_t *)((char *)(h) + offsetof(Histo, off));
#ifdef CL_SYCL_LANGUAGE_VERSION
      stream->memset(off, 0, 4 * Histo::totbins());
#else
      ::memset(off, 0, 4 * Histo::totbins());
#endif
    }

    template <typename Histo>
    inline void launchFinalize(Histo *__restrict__ h,
                               uint8_t *__restrict__ ws
#ifndef CL_SYCL_LANGUAGE_VERSION
                               = cudaStreamDefault
#endif
                               ,
                               sycl::queue *stream
#ifndef CL_SYCL_LANGUAGE_VERSION
                               = cudaStreamDefault
#endif
                               ) try {
#ifdef CL_SYCL_LANGUAGE_VERSION
      assert(ws);
      uint32_t *off = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      size_t wss = Histo::wsSize();
      assert(wss > 0);
      CubDebugExit(cub::DeviceScan::InclusiveSum(ws, wss, off, off, Histo::totbins(), stream));
#else
      h->finalize();
#endif
    } catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename Histo, typename T>
    inline void fillManyFromVector(Histo *__restrict__ h,
                                   uint8_t *__restrict__ ws,
                                   uint32_t nh,
                                   T const *__restrict__ v,
                                   uint32_t const *__restrict__ offsets,
                                   uint32_t totSize,
                                   int nthreads,
                                   sycl::queue *stream
#ifndef CL_SYCL_LANGUAGE_VERSION
                                   = cudaStreamDefault
#endif
    ) {
      launchZero(h, stream);
#ifdef CL_SYCL_LANGUAGE_VERSION
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      /*
      DPCT1049:33: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
      */
      stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)),
            [=](sycl::nd_item<3> item_ct1) { countFromVector(h, nh, v, offsets, item_ct1); });
      });
      launchFinalize(h, ws, stream);
      /*
      DPCT1049:35: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
      */
      stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)),
            [=](sycl::nd_item<3> item_ct1) { fillFromVector(h, nh, v, offsets, item_ct1); });
      });
#else
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
#endif
    }

    template <typename Assoc>
    void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc, sycl::nd_item<3> item_ct1) {
      assoc->bulkFinalizeFill(*apc, item_ct1);
    }

  }  // namespace sycltools
}  // namespace cms

// iteratate over N bins left and right of the one containing "v"
template <typename Hist, typename V, typename Func>
inline __attribute__((always_inline)) void forEachInBins(Hist const &hist, V value, int n, Func func) {
  int bs = Hist::bin(value);
  int be = sycl::min(int(Hist::nbins() - 1), bs + n);
  bs = sycl::max(0, bs - n);
  assert(be >= bs);
  for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
    func(*pj);
  }
}

// iteratate over bins containing all values in window wmin, wmax
template <typename Hist, typename V, typename Func>
inline __attribute__((always_inline)) void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
  auto bs = Hist::bin(wmin);
  auto be = Hist::bin(wmax);
  assert(be >= bs);
  for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
    func(*pj);
  }
}

template <typename T,                  // the type of the discretized input values
          uint32_t NBINS,              // number of bins
          uint32_t SIZE,               // max number of element
          uint32_t S = sizeof(T) * 8,  // number of significant bits in T
          typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
          uint32_t NHISTS = 1     // number of histos stored
          >
class HistoContainer {
public:
  using Counter = uint32_t;

  using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

  using index_type = I;
  using UT = typename std::make_unsigned<T>::type;

  static constexpr uint32_t ilog2(uint32_t v) {
    constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    constexpr uint32_t s[] = {1, 2, 4, 8, 16};

    uint32_t r = 0;  // result of log2(v) will go here
    for (auto i = 4; i >= 0; i--)
      if (v & b[i]) {
        v >>= s[i];
        r |= s[i];
      }
    return r;
  }

  static constexpr uint32_t sizeT() { return S; }
  static constexpr uint32_t nbins() { return NBINS; }
  static constexpr uint32_t nhists() { return NHISTS; }
  static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
  static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
  static constexpr uint32_t capacity() { return SIZE; }

  static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

  static size_t wsSize() {
#ifdef CL_SYCL_LANGUAGE_VERSION
    uint32_t *v = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v, v, totbins());
    return temp_storage_bytes;
#else
    return 0;
#endif
  }

  static constexpr UT bin(T t) {
    constexpr uint32_t shift = sizeT() - nbits();
    constexpr uint32_t mask = (1 << nbits()) - 1;
    return (t >> shift) & mask;
  }

  void zero() {
    for (auto &i : off)
      i = 0;
  }

  void add(CountersOnly const &co) {
    for (uint32_t i = 0; i < totbins(); ++i) {
#ifdef DPCPP_COMPATIBILITY_TEMP
      sycl::atomic<HistoContainer<unsigned int, 8, 8000, 32, unsigned short, 1>::Counter>(
          sycl::global_ptr<HistoContainer<unsigned int, 8, 8000, 32, unsigned short, 1>::Counter>(off + i))
          .fetch_add(co.off[i]);
#else
      auto &a = (std::atomic<Counter> &)(off[i]);
      a += co.off[i];
#endif
    }
  }

  static inline __attribute__((always_inline)) uint32_t atomicIncrement(Counter &x) {
#ifdef DPCPP_COMPATIBILITY_TEMP
    return sycl::atomic<HistoContainer::Counter>(sycl::global_ptr<HistoContainer::Counter>(&x)).fetch_add(1);
#else
    auto &a = (std::atomic<Counter> &)(x);
    return a++;
#endif
  }

  static inline __attribute__((always_inline)) uint32_t atomicDecrement(Counter &x) {
#ifdef DPCPP_COMPATIBILITY_TEMP
    return sycl::atomic<HistoContainer::Counter>(sycl::global_ptr<HistoContainer::Counter>(&x)).fetch_sub(1);
#else
    auto &a = (std::atomic<Counter> &)(x);
    return a--;
#endif
  }

  inline __attribute__((always_inline)) void countDirect(T b) {
    assert(b < nbins());
    atomicIncrement(off[b]);
  }

  inline __attribute__((always_inline)) void fillDirect(T b, index_type j) {
    assert(b < nbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  inline __attribute__((always_inline)) int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
    auto c = apc.add(n);
    if (c.m >= nbins())
      return -int32_t(c.m);
    off[c.m] = c.n;
    for (uint32_t j = 0; j < n; ++j)
      bins[c.n + j] = v[j];
    return c.m;
  }

  inline __attribute__((always_inline)) void bulkFinalize(AtomicPairCounter const &apc) {
    off[apc.get().m] = apc.get().n;
  }

  inline __attribute__((always_inline)) void bulkFinalizeFill(AtomicPairCounter const &apc, sycl::nd_item<3> item_ct1) {
    auto m = apc.get().m;
    auto n = apc.get().n;
    if (m >= nbins()) {  // overflow!
      off[nbins()] = uint32_t(off[nbins() - 1]);
      return;
    }
    auto first = m + item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    for (auto i = first; i < totbins(); i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
      off[i] = n;
    }
  }

  inline __attribute__((always_inline)) void count(T t) {
    uint32_t b = bin(t);
    assert(b < nbins());
    atomicIncrement(off[b]);
  }

  inline __attribute__((always_inline)) void fill(T t, index_type j) {
    uint32_t b = bin(t);
    assert(b < nbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  inline __attribute__((always_inline)) void count(T t, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b < nbins());
    b += histOff(nh);
    assert(b < totbins());
    atomicIncrement(off[b]);
  }

  inline __attribute__((always_inline)) void fill(T t, index_type j, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b < nbins());
    b += histOff(nh);
    assert(b < totbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  inline __attribute__((always_inline)) void finalize(sycl::nd_item<3> item_ct1, Counter *ws = nullptr) {
    assert(off[totbins() - 1] == 0);
    blockPrefixScan(off, totbins(), ws, item_ct1);
    assert(off[totbins() - 1] == off[totbins() - 2]);
  }

  constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
  constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

  constexpr index_type const *begin() const { return bins; }
  constexpr index_type const *end() const { return begin() + size(); }

  constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
  constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

  Counter off[totbins()];
  index_type bins[capacity()];
};

template <typename I,        // type stored in the container (usually an index in a vector of the input values)
          uint32_t MAXONES,  // max number of "ones"
          uint32_t MAXMANYS  // max number of "manys"
          >
using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
