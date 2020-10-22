#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLDataFormats/BeamSpotCUDA.h"
#include "SYCLCore/Product.h"
#include "SYCLDataFormats/SiPixelClustersCUDA.h"
#include "SYCLDataFormats/SiPixelDigisCUDA.h"
#include "SYCLDataFormats/TrackingRecHit2DCUDA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "SYCLCore/ScopedContext.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

class SiPixelRecHitCUDA : public edm::EDProducer {
public:
  explicit SiPixelRecHitCUDA(edm::ProductRegistry& reg);
  ~SiPixelRecHitCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<cms::sycltools::Product<BeamSpotCUDA>> tBeamSpot;
  edm::EDGetTokenT<cms::sycltools::Product<SiPixelClustersCUDA>> token_;
  edm::EDGetTokenT<cms::sycltools::Product<SiPixelDigisCUDA>> tokenDigi_;

  edm::EDPutTokenT<cms::sycltools::Product<TrackingRecHit2DCUDA>> tokenHit_;

  pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitCUDA::SiPixelRecHitCUDA(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<cms::sycltools::Product<BeamSpotCUDA>>()),
      token_(reg.consumes<cms::sycltools::Product<SiPixelClustersCUDA>>()),
      tokenDigi_(reg.consumes<cms::sycltools::Product<SiPixelDigisCUDA>>()),
      tokenHit_(reg.produces<cms::sycltools::Product<TrackingRecHit2DCUDA>>()) {}

void SiPixelRecHitCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& pclusters = iEvent.get(token_);
  cms::sycltools::ScopedContextProduce ctx{pclusters};

  auto const& clusters = ctx.get(pclusters);
  auto const& digis = ctx.get(iEvent, tokenDigi_);
  auto const& bs = ctx.get(iEvent, tBeamSpot);

  auto nHits = clusters.nClusters();
  if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
    std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
  }

  ctx.emplace(iEvent,
              tokenHit_,
              gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.getGPUProductAsync(ctx.stream()), ctx.stream()));
}

DEFINE_FWK_MODULE(SiPixelRecHitCUDA);
