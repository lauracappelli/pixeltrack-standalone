#include <fstream>

#include <CL/sycl.hpp>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"
#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/host_noncached_unique_ptr.h"
#include "SYCLDataFormats/BeamSpotCUDA.h"

class BeamSpotToCUDA : public edm::EDProducer {
public:
  explicit BeamSpotToCUDA(edm::ProductRegistry& reg);
  ~BeamSpotToCUDA() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDPutTokenT<cms::sycltools::Product<BeamSpotCUDA>> bsPutToken_;

  cms::sycltools::host::noncached::unique_ptr<BeamSpotCUDA::Data> bsHost;
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg)
    : bsPutToken_{reg.produces<cms::sycltools::Product<BeamSpotCUDA>>()},
      /*
      DPCT1048:76: The original value cudaHostAllocWriteCombined is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
      */
      bsHost{cms::sycltools::make_host_noncached_unique<BeamSpotCUDA::Data>(0)} {}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  *bsHost = iSetup.get<BeamSpotCUDA::Data>();

  cms::sycltools::ScopedContextProduce ctx{iEvent.streamID()};

  ctx.emplace(iEvent, bsPutToken_, bsHost.get(), ctx.stream());
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
