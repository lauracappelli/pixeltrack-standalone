#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "initialisation.h"

namespace cms::sycltools {

  namespace {

    void syclExceptionHandler(sycl::exception_list exceptions) {
      std::ostringstream msg;
      msg << "Caught asynchronous SYCL exception:";
      for (auto const &exc_ptr : exceptions) {
        try {
          std::rethrow_exception(exc_ptr);
        } catch (cl::sycl::exception const &e) {
          msg << '\n' << e.what();
        }
        throw std::runtime_error(msg.str());
      }
    }

  }

  std::vector<sycl::device> const& enumerateDevices(bool verbose) {
    static const std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::all);
    if (verbose) {
      std::cerr << "Found " << devices.size() << " SYCL devices:" << std::endl;
      for (auto const& device : devices)
        std::cerr << "  - " << device.get_info<cl::sycl::info::device::name>() << std::endl;
      std::cerr << std::endl;
    }
    return devices;
  }

  sycl::device chooseDevice(edm::StreamID id) {
    auto const& devices = enumerateDevices();

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    auto const& device = devices[id % devices.size()];
    return device;
  }

  sycl::queue getDeviceQueue(unsigned int index /* = 0 */) {
    return sycl::queue{chooseDevice(index), syclExceptionHandler, sycl::property::queue::in_order()};
  }

  sycl::queue getDeviceQueue(sycl::device device) {
    return sycl::queue{device, syclExceptionHandler, sycl::property::queue::in_order()};
  }

}  // namespace cms::sycltools
