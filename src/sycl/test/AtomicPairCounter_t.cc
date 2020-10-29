#include <cassert>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/initialisation.h"
#include "SYCLCore/AtomicPairCounter.h"

void update(AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n, sycl::nd_item<3> item) {
  auto i = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);
  if (i >= n)
    return;

  auto m = i % 11;
  m = m % 6 + 1;  // max 6, no 0
  auto c = dc->add(m);
  //assert(c.m < n);
  ind[c.m] = c.n;
  for (auto j = c.n; j < c.n + m; ++j)
    cont[j] = i;
};

void finalize(AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) {
  //assert(dc->get().m == n);
  ind[n] = dc->get().n;
}

void verify(AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n, sycl::nd_item<3> item) {
  auto i = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);
  if (i >= n)
    return;
  //assert(0 == ind[0]);
  //assert(dc->get().m == n);
  //assert(ind[n] == dc->get().n);
  auto ib = ind[i];
  auto ie = ind[i + 1];
  auto k = cont[ib++];
  //assert(k < n);
  //for (; ib < ie; ++ib)
  //assert(cont[ib] == k);
}

#include <iostream>
int main() {
  cms::sycltools::enumerateDevices(true);
  sycl::queue queue = cms::sycltools::getDeviceQueue();

  AtomicPairCounter *dc_d = sycl::malloc_device<AtomicPairCounter>(1, queue);
  queue.memset(dc_d, 0, sizeof(AtomicPairCounter));

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  uint32_t *n_d, *m_d;
  n_d = (uint32_t *)sycl::malloc_device(N * sizeof(int), queue);
  m_d = (uint32_t *)sycl::malloc_device(M * sizeof(int), queue);

  int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int threads = std::min(512, max_work_group_size);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 2000 * threads), sycl::range(1, 1, threads)),
                     [=](sycl::nd_item<3> item) { update(dc_d, n_d, m_d, 10000, item); });
  });
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                     [=](sycl::nd_item<3> item) { finalize(dc_d, n_d, m_d, 10000); });
  });
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 2000 * threads), sycl::range(1, 1, threads)),
                     [=](sycl::nd_item<3> item) { verify(dc_d, n_d, m_d, 10000, item); });
  });

  AtomicPairCounter dc;
  queue.memcpy(&dc, dc_d, sizeof(AtomicPairCounter));

  std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

  return 0;
}
