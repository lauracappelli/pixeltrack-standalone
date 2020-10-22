#ifndef HeterogeneousCore_SYCLCore_chooseDevice_h
#define HeterogeneousCore_SYCLCore_chooseDevice_h

#include "Framework/Event.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id);
}

#endif
