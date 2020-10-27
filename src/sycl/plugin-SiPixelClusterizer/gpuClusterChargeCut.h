#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <CL/sycl.hpp>

#include "SYCLCore/prefixScan.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  void clusterChargeCut(uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
                        uint16_t const* __restrict__ adc,          //  charge of each pixel
                        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
                        uint32_t const* __restrict__ moduleId,     // module id of each module
                        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
                        uint32_t numElements,
                        sycl::nd_item<3> item,
                        sycl::stream stream,
                        int32_t* charge,
                        uint8_t* ok,
                        uint16_t* newclusId,
                        uint16_t* ws) {
    if (item.get_group(2) >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + item.get_group(2)];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);
    assert(thisModuleId == moduleId[item.get_group(2)]);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus == 0)
      return;

    if (item.get_local_id(2) == 0 && nclus > MaxNumClustersPerModules)
      /*
      DPCT1015:183: Output needs adjustment.
      */
      stream << "Warning too many clusters in module %d in block %d: %d > %d\n";

    auto first = firstPixel + item.get_local_id(2);

    if (nclus > MaxNumClustersPerModules) {
      // remove excess  FIXME find a way to cut charge first....
      for (auto i = first; i < numElements; i += item.get_local_range().get(2)) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        if (clusterId[i] >= MaxNumClustersPerModules) {
          id[i] = InvId;
          clusterId[i] = InvId;
        }
      }
      nclus = MaxNumClustersPerModules;
    }

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(2) == 0)
        stream << "start clusterizer for module " << thisModuleId << " in block " << item.get_group(2) << sycl::endl;;
#endif

    assert(nclus <= MaxNumClustersPerModules);
    for (auto i = item.get_local_id(2); i < nclus; i += item.get_local_range().get(2)) {
      charge[i] = 0;
    }
    item.barrier();

    for (auto i = first; i < numElements; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      sycl::atomic<int32_t, sycl::access::address_space::local_space>(sycl::local_ptr<int32_t>(&charge[clusterId[i]]))
          .fetch_add(adc[i]);
    }
    item.barrier();

    auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
    for (auto i = item.get_local_id(2); i < nclus; i += item.get_local_range().get(2)) {
      newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
    }

    item.barrier();

    // renumber

    cms::sycltools::blockPrefixScan(item, newclusId, nclus, ws, stream, 16);

    assert(nclus >= newclusId[nclus - 1]);

    if (nclus == newclusId[nclus - 1])
      return;

    nClustersInModule[thisModuleId] = newclusId[nclus - 1];
    item.barrier();

    // mark bad cluster again
    for (auto i = item.get_local_id(2); i < nclus; i += item.get_local_range().get(2)) {
      if (0 == ok[i])
        newclusId[i] = InvId + 1;
    }
    item.barrier();

    // reassign id
    for (auto i = first; i < numElements; i += item.get_local_range().get(2)) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      clusterId[i] = newclusId[clusterId[i]] - 1;
      if (clusterId[i] == InvId)
        id[i] = InvId;
    }

    //done
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
