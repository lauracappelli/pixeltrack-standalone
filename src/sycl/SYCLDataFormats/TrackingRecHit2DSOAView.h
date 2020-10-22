#ifndef SYCLDataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define SYCLDataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLDataFormats/gpuClusteringConstants.h"
#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/cudaCompat.h"
#include "Geometry/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist = HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;

  __dpct_inline__ uint32_t nHits() const { return m_nHits; }

  __dpct_inline__ float& xLocal(int i) { return m_xl[i]; }
  /*
  DPCT1026:37: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float xLocal(int i) const { return *(m_xl + i); }
  __dpct_inline__ float& yLocal(int i) { return m_yl[i]; }
  /*
  DPCT1026:38: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float yLocal(int i) const { return *(m_yl + i); }

  __dpct_inline__ float& xerrLocal(int i) { return m_xerr[i]; }
  /*
  DPCT1026:39: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float xerrLocal(int i) const { return *(m_xerr + i); }
  __dpct_inline__ float& yerrLocal(int i) { return m_yerr[i]; }
  /*
  DPCT1026:40: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float yerrLocal(int i) const { return *(m_yerr + i); }

  __dpct_inline__ float& xGlobal(int i) { return m_xg[i]; }
  /*
  DPCT1026:41: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float xGlobal(int i) const { return *(m_xg + i); }
  __dpct_inline__ float& yGlobal(int i) { return m_yg[i]; }
  /*
  DPCT1026:42: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float yGlobal(int i) const { return *(m_yg + i); }
  __dpct_inline__ float& zGlobal(int i) { return m_zg[i]; }
  /*
  DPCT1026:43: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float zGlobal(int i) const { return *(m_zg + i); }
  __dpct_inline__ float& rGlobal(int i) { return m_rg[i]; }
  /*
  DPCT1026:44: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ float rGlobal(int i) const { return *(m_rg + i); }

  __dpct_inline__ int16_t& iphi(int i) { return m_iphi[i]; }
  /*
  DPCT1026:45: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ int16_t iphi(int i) const { return *(m_iphi + i); }

  __dpct_inline__ int32_t& charge(int i) { return m_charge[i]; }
  /*
  DPCT1026:46: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ int32_t charge(int i) const { return *(m_charge + i); }
  __dpct_inline__ int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  /*
  DPCT1026:47: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ int16_t clusterSizeX(int i) const { return *(m_xsize + i); }
  __dpct_inline__ int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  /*
  DPCT1026:48: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ int16_t clusterSizeY(int i) const { return *(m_ysize + i); }
  __dpct_inline__ uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  /*
  DPCT1026:49: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ uint16_t detectorIndex(int i) const { return *(m_detInd + i); }

  __dpct_inline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  /*
  DPCT1026:50: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  */
  __dpct_inline__ uint32_t hitsModuleStart(int i) const { return *(m_hitsModuleStart + i); }

  __dpct_inline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __dpct_inline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __dpct_inline__ Hist& phiBinner() { return *m_hist; }
  __dpct_inline__ Hist const& phiBinner() const { return *m_hist; }

  __dpct_inline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __dpct_inline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  int32_t* m_charge;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

#endif
