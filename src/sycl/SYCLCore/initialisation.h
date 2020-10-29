#ifndef SYCLCore_initialise_h
#define SYCLCore_initialise_h

#include <vector>

#include <CL/sycl.hpp>

// for edm::StreamID
#include "Framework/Event.h"

namespace cms::sycltools {

  std::vector<sycl::device> const& enumerateDevices(bool verbose = false);

  sycl::device chooseDevice(edm::StreamID id);

  sycl::queue getDeviceQueue(unsigned int index = 0);

  sycl::queue getDeviceQueue(sycl::device device);

}  // namespace cms::sycltools

#endif  // SYCLCore_initialise_h
