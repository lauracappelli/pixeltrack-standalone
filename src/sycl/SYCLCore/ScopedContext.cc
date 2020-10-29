#include <chrono>
#include <future>

#include <CL/sycl.hpp>

#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/initialisation.h"

namespace cms::sycltools {
  namespace impl {

    ScopedContextBase::ScopedContextBase(edm::StreamID streamID)
        : stream_(cms::sycltools::getDeviceQueue(streamID)) {
          std::cerr << "EDM stream " << streamID << " offload to " << stream_.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
        }

    ScopedContextBase::ScopedContextBase(ProductBase const &data)
        : stream_(data.mayReuseStream()
                      ? data.stream()
                      : cms::sycltools::getDeviceQueue(data.device())) {
          //std::cerr << "EDM stream " << id << " offload to " << stream_.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
        }

    ScopedContextBase::ScopedContextBase(sycl::queue stream) : stream_(std::move(stream)) {}

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(sycl::queue dataStream, bool available, sycl::event dataEvent) {
      if (dataStream.get_device() != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw std::runtime_error("Handling data from multiple devices is not yet supported");
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          dataEvent.wait();
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(sycl::queue stream) {
      std::async([&]() {
        stream.wait();
        waitingTaskHolder_.doneWaiting(nullptr);
      });
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(stream());
    if (contextState_) {
      contextState_->set(stream());
    }
  }

  void ScopedContextAcquire::throwNoState() {
    throw std::runtime_error(
        "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
        "ContextState, but that was not the case");
  }

  ////////////////////

  ScopedContextProduce::~ScopedContextProduce() {
    // the barrier should be a no-op on an ordered queue, but is used to mark the end of the data processing
    event_ = stream().submit_barrier();
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(stream()); }
}  // namespace cms::sycltools
