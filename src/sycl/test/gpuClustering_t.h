#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/device_unique_ptr.h"
// dirty, but works
#include "plugin-SiPixelClusterizer/gpuClustering.h"
#include "plugin-SiPixelClusterizer/gpuClusterChargeCut.h"

using namespace gpuClustering;

int main(void) {
  sycl::queue queue = dpct::get_default_queue();

  int numElements = 256 * 2000;
  // these in reality are already on GPU
  auto h_id = std::make_unique<uint16_t[]>(numElements);
  auto h_x = std::make_unique<uint16_t[]>(numElements);
  auto h_y = std::make_unique<uint16_t[]>(numElements);
  auto h_adc = std::make_unique<uint16_t[]>(numElements);

  auto h_clus = std::make_unique<int[]>(numElements);

  auto d_id = cms::sycltools::make_device_unique<uint16_t[]>(numElements, queue);
  auto d_x = cms::sycltools::make_device_unique<uint16_t[]>(numElements, queue);
  auto d_y = cms::sycltools::make_device_unique<uint16_t[]>(numElements, queue);
  auto d_adc = cms::sycltools::make_device_unique<uint16_t[]>(numElements, queue);
  auto d_clus = cms::sycltools::make_device_unique<int[]>(numElements, queue);
  auto d_moduleStart = cms::sycltools::make_device_unique<uint32_t[]>(MaxNumModules + 1, queue);
  auto d_clusInModule = cms::sycltools::make_device_unique<uint32_t[]>(MaxNumModules, queue);
  auto d_moduleId = cms::sycltools::make_device_unique<uint32_t[]>(MaxNumModules, queue);

  // later random number
  int n = 0;
  int ncl = 0;
  int y[10] = {5, 7, 9, 1, 3, 0, 4, 8, 2, 6};

  auto generateClusters = [&](int kn) {
    auto addBigNoise = 1 == kn % 2;
    if (addBigNoise) {
      constexpr int MaxPixels = 1000;
      int id = 666;
      for (int x = 0; x < 140; x += 3) {
        for (int yy = 0; yy < 400; yy += 3) {
          h_id[n] = id;
          h_x[n] = x;
          h_y[n] = yy;
          h_adc[n] = 1000;
          ++n;
          ++ncl;
          if (MaxPixels <= ncl)
            break;
        }
        if (MaxPixels <= ncl)
          break;
      }
    }

    {
      // isolated
      int id = 42;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = kn == 0 ? 100 : 5000;
      ++n;

      // first column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 0;
      h_adc[n] = 5000;
      ++n;
      // first columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 2;
      h_adc[n] = 5000;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 1;
      h_adc[n] = 5000;
      ++n;

      // last column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 415;
      h_adc[n] = 5000;
      ++n;
      // last columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 415;
      h_adc[n] = 2500;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 414;
      h_adc[n] = 2500;
      ++n;

      // diagonal
      ++ncl;
      for (int x = 20; x < 25; ++x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      // reversed
      for (int x = 45; x > 40; --x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      h_id[n++] = InvId;  // error
      // messy
      int xx[5] = {21, 25, 23, 24, 22};
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 20 + xx[k];
        h_adc[n] = 1000;
        ++n;
      }
      // holes
      ++ncl;
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 100;
        h_adc[n] = kn == 2 ? 100 : 1000;
        ++n;
        if (xx[k] % 2 == 0) {
          h_id[n] = id;
          h_x[n] = xx[k];
          h_y[n] = 101;
          h_adc[n] = 1000;
          ++n;
        }
      }
    }
    {
      // id == 0 (make sure it works!
      int id = 0;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = 5000;
      ++n;
    }
    // all odd id
    for (int id = 11; id <= 1800; id += 2) {
      if ((id / 20) % 2)
        h_id[n++] = InvId;  // error
      for (int x = 0; x < 40; x += 4) {
        ++ncl;
        if ((id / 10) % 2) {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[k];
            h_adc[n] = 100;
            ++n;
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = 1000;
            ++n;
          }
        } else {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[9 - k];
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
            if (y[k] == 3)
              continue;  // hole
            if (id == 51) {
              h_id[n++] = InvId;
              h_id[n++] = InvId;
            }  // error
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
          }
        }
      }
    }
  };  // end lambda
  for (auto kkk = 0; kkk < 5; ++kkk) {
    n = 0;
    ncl = 0;
    generateClusters(kkk);

    std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
    //assert(n <= numElements);

    uint32_t nModules = 0;
    size_t size32 = n * sizeof(unsigned int);
    size_t size16 = n * sizeof(unsigned short);
    // size_t size8 = n * sizeof(uint8_t);

    queue.memcpy(d_moduleStart.get(), &nModules, sizeof(uint32_t)).wait();

    queue.memcpy(d_id.get(), h_id.get(), size16).wait();
    queue.memcpy(d_x.get(), h_x.get(), size16).wait();
    queue.memcpy(d_y.get(), h_y.get(), size16).wait();
    queue.memcpy(d_adc.get(), h_adc.get(), size16).wait();
    // Launch SYCL Kernels
    int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    int threadsPerBlock = std::min(((kkk == 5) ? 512 : ((kkk == 3) ? 128 : 256)), max_work_group_size);
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "SYCL countModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
              << " threads\n";

    queue.submit([&](sycl::handler &cgh) {
      auto d_id_get = d_id.get();
      auto d_moduleStart_get = d_moduleStart.get();
      auto d_clus_get = d_clus.get();

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, blocksPerGrid * threadsPerBlock), sycl::range(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item) { countModules(d_id_get, d_moduleStart_get, d_clus_get, n, item); });
    });

    blocksPerGrid = MaxNumModules;  //nModules;

    std::cout << "SYCL findModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
              << " threads\n";
    queue.memset(d_clusInModule.get(), 0, MaxNumModules * sizeof(uint32_t)).wait();

    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      //auto gMaxHit_ptr = gMaxHit.get_ptr();

      sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::local> msize_acc(cgh);
      sycl::accessor<Hist, 0, sycl::access_mode::read_write, sycl::target::local> hist_acc(cgh);
      sycl::accessor<typename Hist::Counter, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
      sycl::accessor<uint32_t, 0, sycl::access_mode::read_write, sycl::target::local> totGood_acc(cgh);
      sycl::accessor<uint32_t, 0, sycl::access_mode::read_write, sycl::target::local> n40_acc(cgh);
      sycl::accessor<uint32_t, 0, sycl::access_mode::read_write, sycl::target::local> n60_acc(cgh);
      sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::local> n0_acc(cgh);
      sycl::accessor<unsigned int, 0, sycl::access_mode::read_write, sycl::target::local> foundClusters_acc(cgh);

      auto d_id_get = d_id.get();
      auto d_x_get = d_x.get();
      auto d_y_get = d_y.get();
      auto d_moduleStart_get = d_moduleStart.get();
      auto d_clusInModule_get = d_clusInModule.get();
      auto d_moduleId_get = d_moduleId.get();
      auto d_clus_get = d_clus.get();

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, blocksPerGrid * threadsPerBlock), sycl::range(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item) {
            findClus(d_id_get,
                     d_x_get,
                     d_y_get,
                     d_moduleStart_get,
                     d_clusInModule_get,
                     d_moduleId_get,
                     d_clus_get,
                     n,
                     item,
                     out,
                     //gMaxHit_ptr,
                     msize_acc.get_pointer(),
                     hist_acc.get_pointer(),
                     ws_acc.get_pointer(),
                     totGood_acc.get_pointer(),
                     n40_acc.get_pointer(),
                     n60_acc.get_pointer(),
                     n0_acc.get_pointer(),
                     foundClusters_acc.get_pointer());
          });
    });
    queue.wait_and_throw();
    queue.memcpy(&nModules, d_moduleStart.get(), sizeof(uint32_t)).wait();

    uint32_t nclus[MaxNumModules], moduleId[nModules];
    queue.memcpy(&nclus, d_clusInModule.get(), MaxNumModules * sizeof(uint32_t)).wait();

    std::cout << "before charge cut found " << std::accumulate(nclus, nclus + MaxNumModules, 0) << " clusters"
              << std::endl;
    for (auto i = MaxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    if (ncl != std::accumulate(nclus, nclus + MaxNumModules, 0))
      std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::target::local> charge_acc(
          1024 /*MaxNumClustersPerModules*/, cgh);
      sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::target::local> ok_acc(
          1024 /*MaxNumClustersPerModules*/, cgh);
      sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> newclusId_acc(
          1024 /*MaxNumClustersPerModules*/, cgh);
      sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);

      auto d_id_get = d_id.get();
      auto d_adc_get = d_adc.get();
      auto d_moduleStart_get = d_moduleStart.get();
      auto d_clusInModule_get = d_clusInModule.get();
      auto d_moduleId_get = d_moduleId.get();
      auto d_clus_get = d_clus.get();

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, blocksPerGrid * threadsPerBlock), sycl::range(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item) {
            clusterChargeCut(d_id_get,
                             d_adc_get,
                             d_moduleStart_get,
                             d_clusInModule_get,
                             d_moduleId_get,
                             d_clus_get,
                             n,
                             item,
                             out,
                             charge_acc.get_pointer(),
                             ok_acc.get_pointer(),
                             newclusId_acc.get_pointer(),
                             ws_acc.get_pointer());
          });
    });

    queue.wait_and_throw();

    std::cout << "found " << nModules << " Modules active" << std::endl;

    queue.memcpy(h_id.get(), d_id.get(), size16).wait();
    queue.memcpy(h_clus.get(), d_clus.get(), size32).wait();
    queue.memcpy(&nclus, d_clusInModule.get(), MaxNumModules * sizeof(uint32_t)).wait();
    queue.memcpy(&moduleId, d_moduleId.get(), nModules * sizeof(uint32_t)).wait();

    std::set<unsigned int> clids;
    for (int i = 0; i < n; ++i) {
      //assert(h_id[i] != 666);  // only noise
      if (h_id[i] == InvId)
        continue;
      //assert(h_clus[i] >= 0);
      //assert(h_clus[i] < int(nclus[h_id[i]]));
      clids.insert(h_id[i] * 1000 + h_clus[i]);
      // clids.insert(h_clus[i]);
    }

    // verify no hole in numbering
    auto p = clids.begin();
    auto cmid = (*p) / 1000;
    //assert(0 == (*p) % 1000);
    auto c = p;
    ++c;
    std::cout << "first clusters " << *p << ' ' << *c << ' ' << nclus[cmid] << ' ' << nclus[(*c) / 1000] << std::endl;
    std::cout << "last cluster " << *clids.rbegin() << ' ' << nclus[(*clids.rbegin()) / 1000] << std::endl;
    for (; c != clids.end(); ++c) {
      auto cc = *c;
      auto pp = *p;
      auto mid = cc / 1000;
      auto pnc = pp % 1000;
      auto nc = cc % 1000;
      if (mid != cmid) {
        //assert(0 == cc % 1000);
        //assert(nclus[cmid] - 1 == pp % 1000);
        // if (nclus[cmid]-1 != pp%1000) std::cout << "error size " << mid << ": "  << nclus[mid] << ' ' << pp << std::endl;
        cmid = mid;
        p = c;
        continue;
      }
      p = c;
      // assert(nc==pnc+1);
      if (nc != pnc + 1)
        std::cout << "error " << mid << ": " << nc << ' ' << pnc << std::endl;
    }

    std::cout << "found " << std::accumulate(nclus, nclus + MaxNumModules, 0) << ' ' << clids.size() << " clusters"
              << std::endl;
    for (auto i = MaxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    // << " and " << seeds.size() << " seeds" << std::endl;
  }  /// end loop kkk
  return 0;
}
