#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/HistoContainer.h"

template <typename T, int NBINS, int S, int DELTA>
void mykernel(T const *__restrict__ v,
              uint32_t N,
              sycl::nd_item<3> item,
              sycl::stream out,
              sycl::local_ptr<cms::sycltools::HistoContainer<T, NBINS, 12000, S, uint16_t>> hist,
              sycl::local_ptr<typename cms::sycltools::HistoContainer<T, NBINS, 12000, S, uint16_t>::Counter> ws) {
  //assert(v);
  //assert(N == 12000);

  if (item.get_local_id(2) == 0)
    out << "start kernel for %d data\n";

  using Hist = cms::sycltools::HistoContainer<T, NBINS, 12000, S, uint16_t>;

  for (auto j = item.get_local_id(2); j < Hist::totbins(); j += item.get_local_range().get(2)) {
    hist->off[j] = 0;
  }
  item.barrier();

  for (auto j = item.get_local_id(2); j < N; j += item.get_local_range().get(2))
    hist->count(v[j]);
  item.barrier();

  //assert(0 == hist->size());
  item.barrier();

  hist->finalize(item, ws, out);
  item.barrier();

  //assert(N == hist->size());
  for (auto j = item.get_local_id(2); j < Hist::nbins(); j += item.get_local_range().get(2))
    //assert(hist->off[j] <= hist->off[j + 1]);
    item.barrier();

  if (item.get_local_id(2) < 32)
    ws[item.get_local_id(2)] = 0;  // used by prefix scan...
  item.barrier();

  for (auto j = item.get_local_id(2); j < N; j += item.get_local_range().get(2))
    hist->fill(v[j], j);
  item.barrier();
  //assert(0 == hist->off[0]);
  //assert(N == hist->size());

  for (auto j = item.get_local_id(2); j < hist->size() - 1; j += item.get_local_range().get(2)) {
    auto p = hist->begin() + j;
    //assert((*p) < N);
    auto k1 = Hist::bin(v[*p]);
    auto k2 = Hist::bin(v[*(p + 1)]);
    //assert(k2 >= k1);
  }

  for (auto i = item.get_local_id(2); i < hist->size(); i += item.get_local_range().get(2)) {
    auto p = hist->begin() + i;
    auto j = *p;
    auto b0 = Hist::bin(v[j]);
    int tot = 0;
    auto ftest = [&](int k) {
      //assert(k >= 0 && k < (int) N);
      ++tot;
    };
    forEachInWindow(*hist, v[j], v[j], ftest);
    int rtot = hist->size(b0);
    //assert(tot == rtot);
    tot = 0;
    auto vm = int(v[j]) - DELTA;
    auto vp = int(v[j]) + DELTA;
    constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
    vm = sycl::max(vm, 0);
    vm = sycl::min(vm, vmax);
    vp = sycl::min(vp, vmax);
    vp = sycl::max(vp, 0);
    //assert(vp >= vm);
    forEachInWindow(*hist, vm, vp, ftest);
    int bp = Hist::bin(vp);
    int bm = Hist::bin(vm);
    rtot = hist->end(bp) - hist->begin(bm);
    //assert(tot == rtot);
  }
}

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go() {
  sycl::queue queue = dpct::get_default_queue();

  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;
  T v[N];

  auto v_d = cms::sycltools::make_device_unique<T[]>(N, queue);
  //assert(v_d.get());

  using Hist = cms::sycltools::HistoContainer<T, NBINS, N, S, uint16_t>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    //assert(v_d.get());
    //assert(v);
    queue.memcpy(v_d.get(), v, N * sizeof(T)).wait();
    //assert(v_d.get());
    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      sycl::accessor<Hist, 0, sycl::access_mode::read_write, sycl::target::local> hist_acc(cgh);
      sycl::accessor<typename Hist::Counter, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);

      auto v_d_get = v_d.get();

      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 256), sycl::range(1, 1, 256)), [=](sycl::nd_item<3> item) {
        mykernel<T, NBINS, S, DELTA>(v_d_get, N, item, out, hist_acc.get_pointer(), ws_acc.get_pointer());
      });
    });
  }
}

int main() {
  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
