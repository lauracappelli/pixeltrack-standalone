#ifndef SYCLDataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define SYCLDataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/cudaCompat.h"

class SiPixelDigisCUDA {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, sycl::queue *stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  uint16_t *xx() { return xx_d.get(); }
  uint16_t *yy() { return yy_d.get(); }
  uint16_t *adc() { return adc_d.get(); }
  uint16_t *moduleInd() { return moduleInd_d.get(); }
  int32_t *clus() { return clus_d.get(); }
  uint32_t *pdigi() { return pdigi_d.get(); }
  uint32_t *rawIdArr() { return rawIdArr_d.get(); }

  uint16_t const *xx() const { return xx_d.get(); }
  uint16_t const *yy() const { return yy_d.get(); }
  uint16_t const *adc() const { return adc_d.get(); }
  uint16_t const *moduleInd() const { return moduleInd_d.get(); }
  int32_t const *clus() const { return clus_d.get(); }
  uint32_t const *pdigi() const { return pdigi_d.get(); }
  uint32_t const *rawIdArr() const { return rawIdArr_d.get(); }

  uint16_t const *c_xx() const { return xx_d.get(); }
  uint16_t const *c_yy() const { return yy_d.get(); }
  uint16_t const *c_adc() const { return adc_d.get(); }
  uint16_t const *c_moduleInd() const { return moduleInd_d.get(); }
  int32_t const *c_clus() const { return clus_d.get(); }
  uint32_t const *c_pdigi() const { return pdigi_d.get(); }
  uint32_t const *c_rawIdArr() const { return rawIdArr_d.get(); }

  cms::sycltools::host::unique_ptr<uint16_t[]> adcToHostAsync(sycl::queue *stream) const;
  cms::sycltools::host::unique_ptr<int32_t[]> clusToHostAsync(sycl::queue *stream) const;
  cms::sycltools::host::unique_ptr<uint32_t[]> pdigiToHostAsync(sycl::queue *stream) const;
  cms::sycltools::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(sycl::queue *stream) const;

  class DeviceConstView {
  public:
    // DeviceConstView() = default;

    /*
    DPCT1026:14: The call to __ldg was removed, because there is no correspoinding API in DPC++.
    */
    __dpct_inline__ uint16_t xx(int i) const { return *(xx_ + i); }
    /*
    DPCT1026:15: The call to __ldg was removed, because there is no correspoinding API in DPC++.
    */
    __dpct_inline__ uint16_t yy(int i) const { return *(yy_ + i); }
    /*
    DPCT1026:16: The call to __ldg was removed, because there is no correspoinding API in DPC++.
    */
    __dpct_inline__ uint16_t adc(int i) const { return *(adc_ + i); }
    /*
    DPCT1026:17: The call to __ldg was removed, because there is no correspoinding API in DPC++.
    */
    __dpct_inline__ uint16_t moduleInd(int i) const { return *(moduleInd_ + i); }
    /*
    DPCT1026:18: The call to __ldg was removed, because there is no correspoinding API in DPC++.
    */
    __dpct_inline__ int32_t clus(int i) const { return *(clus_ + i); }

    friend class SiPixelDigisCUDA;

    // private:
    uint16_t const *xx_;
    uint16_t const *yy_;
    uint16_t const *adc_;
    uint16_t const *moduleInd_;
    int32_t const *clus_;
  };

  const DeviceConstView *view() const { return view_d.get(); }

private:
  // These are consumed by downstream device code
  cms::sycltools::device::unique_ptr<uint16_t[]> xx_d;         // local coordinates of each pixel
  cms::sycltools::device::unique_ptr<uint16_t[]> yy_d;         //
  cms::sycltools::device::unique_ptr<uint16_t[]> adc_d;        // ADC of each pixel
  cms::sycltools::device::unique_ptr<uint16_t[]> moduleInd_d;  // module id of each pixel
  cms::sycltools::device::unique_ptr<int32_t[]> clus_d;        // cluster id of each pixel
  cms::sycltools::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer

  // These are for CPU output; should we (eventually) place them to a
  // separate product?
  cms::sycltools::device::unique_ptr<uint32_t[]> pdigi_d;
  cms::sycltools::device::unique_ptr<uint32_t[]> rawIdArr_d;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif
