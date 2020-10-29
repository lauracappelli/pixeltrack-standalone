#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "Geometry/phase1PixelTopology.h"
#include "SYCLCore/HistoContainer.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  void countModules(uint16_t const* __restrict__ id,
                    uint32_t* __restrict__ moduleStart,
                    int32_t* __restrict__ clusterId,
                    int numElements,
                    sycl::nd_item<3> item) {
    int first = item.get_local_range().get(2) * item.get_group(2) + item.get_local_id(2);
    for (int i = first; i < numElements; i += item.get_group_range(2) * item.get_local_range().get(2)) {
      clusterId[i] = i;
      if (InvId == id[i])
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == InvId)
        --j;
      if (j < 0 or id[j] != id[i]) {
        // boundary...
        auto loc = dpct::atomic_fetch_compare_inc(moduleStart, MaxNumModules);
        moduleStart[loc + 1] = i;
      }
    }
  }

  /*
   * DPCT moved the shared data outside of the function, so we need to make all types available
   * in the function prototype as well.
   */
  //init hist  (ymax=416 < 512 : 9bits)
  constexpr uint32_t maxPixInModule = 4000;
  constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
  using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

  //  __launch_bounds__(256,4)
  void findClus(uint16_t const* __restrict__ id,           // module id of each pixel
                uint16_t const* __restrict__ x,            // local coordinates of each pixel
                uint16_t const* __restrict__ y,            //
                uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
                uint32_t* __restrict__ moduleId,           // output: module id of each module
                int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
                int numElements,
                sycl::nd_item<3> item,
                sycl::stream out,
                int* msize,
                Hist* hist,
                typename Hist::Counter* ws,
                uint32_t* totGood,
                uint32_t* n40,
                uint32_t* n60,
                int* n0,
                unsigned int* foundClusters) {
    if (item.get_group(2) >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + item.get_group(2)];
    auto thisModuleId = id[firstPixel];
    //assert(thisModuleId < MaxNumModules);

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(2) == 0)
        out << "start clusterizer for module " << thisModuleId << " in block " << item.get_local_id(2) << sycl::endl;
#endif

    auto first = firstPixel + item.get_local_id(2);

    // find the index of the first pixel not belonging to this module (or invalid)

    *msize = numElements;
    item.barrier();

    // skip threads not associated to an existing pixel
    for (int i = first; i < numElements; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  // find the first pixel in a different module
        sycl::atomic<int, sycl::access::address_space::local_space>(sycl::local_ptr<int>(msize)).fetch_min(i);
        break;
      }
    }

    for (auto j = item.get_local_id(2); j < Hist::totbins(); j += item.get_local_range().get(2)) {
      hist->off[j] = 0;
    }
    item.barrier();

    //assert((*msize == numElements) or ((*msize < numElements) and (id[(*msize)] != thisModuleId)));

    // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
    if (0 == item.get_local_id(2)) {
      if (*msize - firstPixel > maxPixInModule) {
        out << "too many pixels in module %d: %d > %d\n";
        *msize = maxPixInModule + firstPixel;
      }
    }

    item.barrier();
    //assert(*msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
    *totGood = 0;
    item.barrier();
#endif

    // fill histo
    for (int i = first; i < *msize; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->count(y[i]);
#ifdef GPU_DEBUG
      sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(totGood)).fetch_add(1);
#endif
    }
    item.barrier();
    if (item.get_local_id(2) < 32)
      ws[item.get_local_id(2)] = 0;  // used by prefix scan...
    item.barrier();
    hist->finalize(item, ws, out);
    item.barrier();
#ifdef GPU_DEBUG
    //assert(hist->size() == *totGood);
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(2) == 0)
        out << "histo size %d\n";
#endif
    for (int i = first; i < *msize; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->fill(y[i], i - firstPixel);
    }

    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations
    constexpr int maxiter = 16;
    // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
    constexpr int maxNeighbours = 10;
    //assert((hist->size() / item.get_local_range().get(2)) <= maxiter);
    // nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;

    item.barrier();  // for hit filling!

#ifdef GPU_DEBUG
    // look for anomalous high occupancy

    *n40 = *n60 = 0;
    item.barrier();
    for (auto j = item.get_local_id(2); j < Hist::nbins(); j += item.get_local_range().get(2)) {
      if (hist->size(j) > 60)
        sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(n60)).fetch_add(1);
      if (hist->size(j) > 40)
        sycl::atomic<uint32_t, sycl::access::address_space::local_space>(sycl::local_ptr<uint32_t>(n40)).fetch_add(1);
    }
    item.barrier();
    if (0 == item.get_local_id(2)) {
      if (*n60 > 0)
        out << "columns with more than 60 px %d in %d\n";
      else if (*n40 > 0)
        out << "columns with more than 40 px %d in %d\n";
    }
    item.barrier();
#endif

    // fill NN
    for (auto j = item.get_local_id(2), k = 0UL; j < hist->size(); j += item.get_local_range().get(2), ++k) {
      //assert(k < maxiter);
      auto p = hist->begin() + j;
      auto i = *p + firstPixel;
      //assert(id[i] != InvId);
      //assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist->end(be);
      ++p;
      //assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        //assert(m != i);
        //assert(int(y[m]) - int(y[i]) >= 0);
        //assert(int(y[m]) - int(y[i]) <= 1);
        if (sycl::abs(int(x[m]) - int(x[i])) > 1)
          continue;
        auto l = nnn[k]++;
        //assert(l < maxNeighbours);
        nn[k][l] = *p;
      }
    }

    // for each pixel, look at all the pixels until the end of the module;
    // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
    // after the loop, all the pixel in each cluster should have the id equeal to the lowest
    // pixel in the cluster ( clus[i] == i ).
    bool more = true;
    int nloops = 0;
    while ((item.barrier(), sycl::ONEAPI::any_of(item.get_group(), more))) {
      if (1 == nloops % 2) {
        for (auto j = item.get_local_id(2), k = 0UL; j < hist->size(); j += item.get_local_range().get(2), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (auto j = item.get_local_id(2), k = 0UL; j < hist->size(); j += item.get_local_range().get(2), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            //assert(m != i);
            auto old = sycl::atomic<int32_t>(sycl::global_ptr<int32_t>(&clusterId[m])).fetch_min(clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            sycl::atomic<int32_t>(sycl::global_ptr<int32_t>(&clusterId[i])).fetch_min(old);
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

#ifdef GPU_DEBUG
    {
      if (item.get_local_id(2) == 0)
        *n0 = nloops;
      item.barrier();
      auto ok = *n0 == nloops;
      //assert((item.barrier(), sycl::intel::all_of(item.get_group(), ok)));
      if (thisModuleId % 100 == 1)
        if (item.get_local_id(2) == 0)
          out << "# loops %d\n";
    }
#endif

    *foundClusters = 0;
    item.barrier();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = first; i < *msize; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] == i) {
        auto old = dpct::atomic_fetch_compare_inc<sycl::access::address_space::local_space>(foundClusters, 0xffffffff);
        clusterId[i] = -(old + 1);
      }
    }
    item.barrier();

    // propagate the negative id to all the pixels in the cluster.
    for (int i = first; i < *msize; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] >= 0) {
        // mark each pixel in a cluster with the same id as the first one
        clusterId[i] = clusterId[clusterId[i]];
      }
    }
    item.barrier();

    // adjust the cluster id to be a positive value starting from 0
    for (int i = first; i < *msize; i += item.get_local_range().get(2)) {
      if (id[i] == InvId) {  // skip invalid pixels
        clusterId[i] = -9999;
        continue;
      }
      clusterId[i] = -clusterId[i] - 1;
    }
    item.barrier();

    if (item.get_local_id(2) == 0) {
      nClustersInModule[thisModuleId] = *foundClusters;
      moduleId[item.get_group(2)] = thisModuleId;
#ifdef GPU_DEBUG
      // FIXME
      // forget about using atomics in SYCL
      if (thisModuleId % 100 == 1)
        out << "%d clusters in module %d\n";
#endif
    }
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
