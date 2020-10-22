#ifndef HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h

#include <optional>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    /**
   * Returns true if the work captured by the event (= queued to the
   * SYCL queue when the event was obtained) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(sycl::event const& event) {
      return event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
    }

    inline bool eventWorkHasCompleted(std::optional<sycl::event> const& event) {
      return event.has_value() and eventWorkHasCompleted(event.value());
    }
  }  // namespace sycltools
}  // namespace cms

#endif
