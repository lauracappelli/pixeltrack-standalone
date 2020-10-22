#ifndef HeterogeneousCore_SYCLCore_ESProduct_h
#define HeterogeneousCore_SYCLCore_ESProduct_h

#include <atomic>
#include <cassert>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <CL/sycl.hpp>

#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/eventWorkHasCompleted.h"

namespace cms {
  namespace sycltools {
    template <typename T>
    class ESProduct {
    public:
      ESProduct() {
        auto const& devices = enumerateDevices();
        for (auto const& device : devices) {
          // add a default constructed object to the map
          gpuDataPerDevice_[device];
        }
        assert(devices.size() == gpuDataPerDevice_.size());
      }

      ~ESProduct() = default;

      // The current device is the one associated to the `stream` SYCL queue.
      // `transferAsync` should be a function of (T&, sycl::queue)
      // that enqueues asynchronous transfers (possibly kernels as well) to the SYCL queue.
      template <typename F>
      const T& dataForCurrentDeviceAsync(sycl::queue stream, F transferAsync) const {
        auto& data = gpuDataPerDevice_.at(stream.get_device());

        // If GPU data has already been filled, we can return it
        // immediately
        if (not data.m_filled.load()) {
          // It wasn't, so need to fill it
          std::scoped_lock<std::mutex> lk{data.m_mutex};

          if (data.m_filled.load()) {
            // Other thread marked it filled while we were locking the mutex, so we're free to return it
            return data.m_data;
          }

          if (data.m_fillingStream) {
            // Someone else is filling

            // Check first if the recorded event has occurred
            if (eventWorkHasCompleted(data.m_event)) {
              // It was, so data is accessible from all SYCL queues on
              // the device. Set the 'filled' for all subsequent calls and
              // return the value
              auto should_be_false = data.m_filled.exchange(true);
              assert(not should_be_false);
              data.m_fillingStream.reset();
            } else if (data.m_fillingStream != stream) {
              // Filling is still going on, in a different SYCL queue.
              // Submit a barrier to our queae and return the value.
              // Subsequent work in our queue will wait for the event to occur
              // (i.e. for the transfer to finish).
              stream.submit_barrier({data.m_event.value()});
            }
            // Filling is still going on, in the same SYCL queue.
            // Return the value immediately.
            // Subsequent work in our queue will anyway wait for the
            // transfer to finish.
          } else {
            // Now we can be sure that the data is not yet on the GPU, and
            // this thread is the first to try that.
            transferAsync(data.m_data, stream);
            assert(not data.m_fillingStream);
            data.m_fillingStream = stream;
            // Now the filling has been enqueued to the stream, so we
            // can return the GPU data immediately, since all subsequent
            // work must be either enqueued to the stream, or the stream
            // must be synchronized by the caller
          }
        }

        return data.m_data;
      }

    private:
      struct Item {
        mutable std::mutex m_mutex;
        mutable std::optional<sycl::event> m_event;  // guarded by m_mutex
        mutable std::optional<sycl::queue>
            m_fillingStream;                         // guarded by m_mutex, non-empty when a thread is copying the data
        mutable std::atomic<bool> m_filled = false;  // fast check if data has been already filled or not
        mutable T m_data;                            // guarded by m_mutex
      };

      std::unordered_map<sycl::device, Item> gpuDataPerDevice_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif
