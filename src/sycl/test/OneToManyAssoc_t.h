#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#ifdef CL_SYCL_LANGUAGE_VERSION
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/currentDevice.h"
#endif

#include "SYCLCore/HistoContainer.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;
using Assoc = OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;

using SmallAssoc = OneToManyAssoc<uint16_t, 128, MaxAssocs>;

using Multiplicity = OneToManyAssoc<uint16_t, 8, MaxTk>;

using TK = std::array<uint16_t, 4>;

void countMultiLocal(TK const* __restrict__ tk,
                     Multiplicity* __restrict__ assoc,
                     int32_t n,
                     sycl::nd_item<3> item_ct1,
                     Multiplicity::CountersOnly* local) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (int i = first; i < n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    if (item_ct1.get_local_id(2) == 0)
      local->zero();
    item_ct1.barrier();
    local->countDirect(2 + i % 4);
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0)
      assoc->add(*local);
  }
}

void countMulti(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n, sycl::nd_item<3> item_ct1) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (int i = first; i < n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2))
    assoc->countDirect(2 + i % 4);
}

void verifyMulti(Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2, sycl::nd_item<3> item_ct1) {
  auto first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (auto i = first; i < Multiplicity::totbins();
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2))
    assert(m1->off[i] == m2->off[i]);
}

void count(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n, sycl::nd_item<3> item_ct1) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (int i = first; i < 4 * n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->countDirect(tk[k][j]);
  }
}

void fill(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n, sycl::nd_item<3> item_ct1) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (int i = first; i < 4 * n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->fillDirect(tk[k][j], k);
  }
}

void verify(Assoc* __restrict__ assoc) { assert(assoc->size() < Assoc::capacity()); }

template <typename Assoc>
void fillBulk(AtomicPairCounter* apc,
              TK const* __restrict__ tk,
              Assoc* __restrict__ assoc,
              int32_t n,
              sycl::nd_item<3> item_ct1) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  for (int k = first; k < n; k += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    auto m = tk[k][3] < MaxElem ? 4 : 3;
    assoc->bulkFill(*apc, &tk[k][0], m);
  }
}

template <typename Assoc>
void verifyBulk(Assoc const* __restrict__ assoc, AtomicPairCounter const* apc, sycl::stream stream_ct1) {
  if (apc->get().m >= Assoc::nbins())
    /*
    DPCT1015:234: Output needs adjustment.
    */
    stream_ct1 << "Overflow %d %d\n";
  assert(assoc->size() < Assoc::capacity());
}

int main() {
#ifdef CL_SYCL_LANGUAGE_VERSION
  auto current_device = cms::cuda::currentDevice();
#else
  // make sure cuda emulation is working
  std::cout << "cuda x's " << threadIdx.x << ' ' << blockIdx.x << ' ' << blockDim.x << ' ' << gridDim.x << std::endl;
  std::cout << "cuda y's " << threadIdx.y << ' ' << blockIdx.y << ' ' << blockDim.y << ' ' << gridDim.y << std::endl;
  std::cout << "cuda z's " << threadIdx.z << ' ' << blockIdx.z << ' ' << blockDim.z << ' ' << gridDim.z << std::endl;
  assert(threadIdx.x == 0);
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  assert(blockIdx.x == 0);
  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(blockDim.x == 1);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
#endif

  std::cout << "OneToManyAssoc " << Assoc::nbins() << ' ' << Assoc::capacity() << ' ' << Assoc::wsSize() << std::endl;
  std::cout << "OneToManyAssoc (small) " << SmallAssoc::nbins() << ' ' << SmallAssoc::capacity() << ' '
            << SmallAssoc::wsSize() << std::endl;

  std::mt19937 eng;

  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  std::vector<std::array<uint16_t, 4>> tr(N);

  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

#ifdef CL_SYCL_LANGUAGE_VERSION
  auto v_d = cms::cuda::make_device_unique<std::array<uint16_t, 4>[]>(N, nullptr);
  assert(v_d.get());
  auto a_d = cms::cuda::make_device_unique<Assoc[]>(1, nullptr);
  auto sa_d = cms::cuda::make_device_unique<SmallAssoc[]>(1, nullptr);
  auto ws_d = cms::cuda::make_device_unique<uint8_t[]>(Assoc::wsSize(), nullptr);

  /*
  DPCT1003:235: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memcpy(v_d.get(), tr.data(), N * sizeof(std::array<uint16_t, 4>)).wait(), 0));
#else
  auto a_d = std::make_unique<Assoc>();
  auto sa_d = std::make_unique<SmallAssoc>();
  auto v_d = tr.data();
#endif

  cms::cuda::launchZero(a_d.get(), 0);

#ifdef CL_SYCL_LANGUAGE_VERSION
  auto nThreads = 256;
  auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  /*
  DPCT1049:236: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto v_d_get_ct0 = v_d.get();
    auto a_d_get_ct1 = a_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { count(v_d_get_ct0, a_d_get_ct1, N, item_ct1); });
  });

  cms::cuda::launchFinalize(a_d.get(), ws_d.get(), 0);
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto a_d_get_ct0 = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) { verify(a_d_get_ct0); });
  });
  /*
  DPCT1049:237: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto v_d_get_ct0 = v_d.get();
    auto a_d_get_ct1 = a_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { fill(v_d_get_ct0, a_d_get_ct1, N, item_ct1); });
  });
#else
  count(v_d, a_d.get(), N);
  cms::cuda::launchFinalize(a_d.get());
  verify(a_d.get());
  fill(v_d, a_d.get(), N);
#endif

  Assoc la;

#ifdef CL_SYCL_LANGUAGE_VERSION
  /*
  DPCT1003:238: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memcpy(&la, a_d.get(), sizeof(Assoc)).wait(), 0));
#else
  memcpy(&la, a_d.get(), sizeof(Assoc));  // not required, easier
#endif

  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = la.size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  AtomicPairCounter* dc_d;
  AtomicPairCounter dc(0);

#ifdef CL_SYCL_LANGUAGE_VERSION
  /*
  DPCT1003:239: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dc_d = sycl::malloc_device<AtomicPairCounter>(1, dpct::get_default_queue()), 0));
  /*
  DPCT1003:240: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memset(dc_d, 0, sizeof(AtomicPairCounter)).wait(), 0));
  nBlocks = (N + nThreads - 1) / nThreads;
  /*
  DPCT1049:241: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto v_d_get_ct1 = v_d.get();
    auto a_d_get_ct2 = a_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { fillBulk(dc_d, v_d_get_ct1, a_d_get_ct2, N, item_ct1); });
  });
  /*
  DPCT1049:242: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto a_d_get_ct1 = a_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { finalizeBulk(dc_d, a_d_get_ct1, item_ct1); });
  });
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    auto a_d_get_ct0 = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) { verifyBulk(a_d_get_ct0, dc_d, stream_ct1); });
  });

  /*
  DPCT1003:243: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memcpy(&la, a_d.get(), sizeof(Assoc)).wait(), 0));
  /*
  DPCT1003:244: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memcpy(&dc, dc_d, sizeof(AtomicPairCounter)).wait(), 0));

  /*
  DPCT1003:245: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memset(dc_d, 0, sizeof(AtomicPairCounter)).wait(), 0));
  /*
  DPCT1049:246: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto v_d_get_ct1 = v_d.get();
    auto sa_d_get_ct2 = sa_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { fillBulk(dc_d, v_d_get_ct1, sa_d_get_ct2, N, item_ct1); });
  });
  /*
  DPCT1049:247: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto sa_d_get_ct1 = sa_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { finalizeBulk(dc_d, sa_d_get_ct1, item_ct1); });
  });
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    auto sa_d_get_ct0 = sa_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) { verifyBulk(sa_d_get_ct0, dc_d, stream_ct1); });
  });

#else
  dc_d = &dc;
  fillBulk(dc_d, v_d, a_d.get(), N);
  cms::cuda::finalizeBulk(dc_d, a_d.get());
  verifyBulk(a_d.get(), dc_d);
  memcpy(&la, a_d.get(), sizeof(Assoc));

  AtomicPairCounter sdc(0);
  fillBulk(&sdc, v_d, sa_d.get(), N);
  cms::cuda::finalizeBulk(&sdc, sa_d.get());
  verifyBulk(sa_d.get(), &sdc);

#endif

  std::cout << "final counter value " << dc.get().n << ' ' << dc.get().m << std::endl;

  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = la.size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
#ifdef CL_SYCL_LANGUAGE_VERSION
  auto m1_d = cms::cuda::make_device_unique<Multiplicity[]>(1, nullptr);
  auto m2_d = cms::cuda::make_device_unique<Multiplicity[]>(1, nullptr);
#else
  auto m1_d = std::make_unique<Multiplicity>();
  auto m2_d = std::make_unique<Multiplicity>();
#endif
  cms::cuda::launchZero(m1_d.get(), 0);
  cms::cuda::launchZero(m2_d.get(), 0);

#ifdef CL_SYCL_LANGUAGE_VERSION
  nBlocks = (4 * N + nThreads - 1) / nThreads;
  /*
  DPCT1049:248: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto v_d_get_ct0 = v_d.get();
    auto m1_d_get_ct1 = m1_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) { countMulti(v_d_get_ct0, m1_d_get_ct1, N, item_ct1); });
  });
  /*
  DPCT1049:249: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    sycl::accessor<Multiplicity::CountersOnly, 0, sycl::access::mode::read_write, sycl::access::target::local>
        local_acc_ct1(cgh);

    auto v_d_get_ct0 = v_d.get();
    auto m2_d_get_ct1 = m2_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, nBlocks) * sycl::range(1, 1, nThreads), sycl::range(1, 1, nThreads)),
        [=](sycl::nd_item<3> item_ct1) {
          countMultiLocal(v_d_get_ct0, m2_d_get_ct1, N, item_ct1, local_acc_ct1.get_pointer());
        });
  });
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto m1_d_get_ct0 = m1_d.get();
    auto m2_d_get_ct1 = m2_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, Multiplicity::totbins()), sycl::range(1, 1, Multiplicity::totbins())),
        [=](sycl::nd_item<3> item_ct1) { verifyMulti(m1_d_get_ct0, m2_d_get_ct1, item_ct1); });
  });

  cms::cuda::launchFinalize(m1_d.get(), ws_d.get(), 0);
  cms::cuda::launchFinalize(m2_d.get(), ws_d.get(), 0);
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    auto m1_d_get_ct0 = m1_d.get();
    auto m2_d_get_ct1 = m2_d.get();

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, Multiplicity::totbins()), sycl::range(1, 1, Multiplicity::totbins())),
        [=](sycl::nd_item<3> item_ct1) { verifyMulti(m1_d_get_ct0, m2_d_get_ct1, item_ct1); });
  });

  /*
  DPCT1010:250: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
  /*
  DPCT1003:251: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
#else
  countMulti(v_d, m1_d.get(), N);
  countMultiLocal(v_d, m2_d.get(), N);
  verifyMulti(m1_d.get(), m2_d.get());

  cms::cuda::launchFinalize(m1_d.get());
  cms::cuda::launchFinalize(m2_d.get());
  verifyMulti(m1_d.get(), m2_d.get());
#endif
  return 0;
}
