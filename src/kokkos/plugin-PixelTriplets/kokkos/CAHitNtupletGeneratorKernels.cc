#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

namespace KOKKOS_NAMESPACE {
  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv,
                                                       Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                       KokkosExecSpace const &execSpace) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitContainer::capacity()), KOKKOS_LAMBDA(size_t i) {
          kernel_fillHitDetIndices(&(tracks_d().hitIndices), hv, &(tracks_d().detIndices), i);
        });
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh,
                                                   Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                   KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d().hitIndices;
    auto *quality_d = (Quality *)(&tracks_d().m_quality);

    // zero tuples
    cms::kokkos::launchZero(tuples_d, execSpace);

    auto nhits = hh.nHits();
    assert(nhits <= pixelGPUConstants::maxNumberOfHits);

    // std::cout << "N hits " << nhits << std::endl;
    // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

    int nthTot = 64;
    int stride = 4;
    int teamSize = nthTot / stride;
    int leagueSize = (3 * m_params.maxNumberOfDoublets_ / 4 + teamSize - 1) / teamSize;
    int rescale = leagueSize / 65536;
    teamSize *= (rescale + 1);
    leagueSize = (3 * m_params.maxNumberOfDoublets_ / 4 + teamSize - 1) / teamSize;
    assert(leagueSize < 65536);
    assert(teamSize > 0 && 0 == teamSize % 16);
    teamSize *= stride;

#ifdef KOKKOS_BACKEND_SERIAL
    Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, 1};
    // unit stride loop for serial execution
    stride = 1;
#else
    Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, teamSize};
#endif
    const auto *hhp = hh.view();

    // Kokkos::View as local variables to pass to the lambda
    auto d_hitTuple_apc_ = device_hitTuple_apc_;
    auto d_hitToTuple_apc_ = device_hitToTuple_apc_;
    auto d_theCells_ = device_theCells_;
    auto d_nCells_ = device_nCells_;
    auto d_theCellNeighbors_ = device_theCellNeighbors_;
    auto d_theCellTracks_ = device_theCellTracks_;
    auto d_isOuterHitOfCell_ = device_isOuterHitOfCell_;
    auto d_tupleMultiplicity_ = device_tupleMultiplicity_;

    {
      // capturing this by the lambda leads to illegal memory access with CUDA
      auto const hardCurvCut = m_params.hardCurvCut_;
      auto const ptmin = m_params.ptmin_;
      auto const CAThetaCutBarrel = m_params.CAThetaCutBarrel_;
      auto const CAThetaCutForward = m_params.CAThetaCutForward_;
      auto const dcaCutInnerTriplet = m_params.dcaCutInnerTriplet_;
      auto const dcaCutOuterTriplet = m_params.dcaCutOuterTriplet_;
      Kokkos::parallel_for(
          "kernel_connect", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            kernel_connect(d_hitTuple_apc_,
                           d_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                           hhp,
                           d_theCells_,
                           d_nCells_,
                           d_theCellNeighbors_,  // not used at the moment
                           d_isOuterHitOfCell_,
                           hardCurvCut,
                           ptmin,
                           CAThetaCutBarrel,
                           CAThetaCutForward,
                           dcaCutInnerTriplet,
                           dcaCutOuterTriplet,
                           stride,
                           teamMember);
          });
    }

    if (nhits > 1 && m_params.earlyFishbone_) {
      int teamSize = 128;
      int stride = 16;
      int blockSize = teamSize / stride;
      int leagueSize = (nhits + blockSize - 1) / blockSize;
#ifdef KOKKOS_BACKEND_SERIAL
      Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, 1};
      // unit stride loop for serial execution
      stride = 1;
#else
      Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, teamSize};
#endif
      Kokkos::parallel_for(
          "earlyfishbone", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            gpuPixelDoublets::fishbone(
                hhp, d_theCells_, d_nCells_, d_isOuterHitOfCell_, nhits, false, stride, teamMember);
          });
    }

    {
      auto const minHitsPerNtuplet = m_params.minHitsPerNtuplet_;
      Kokkos::parallel_for(
          "kernel_find_ntuplets",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < d_nCells_()) {
              kernel_find_ntuplets(
                  hhp, d_theCells_, d_theCellTracks_, tuples_d, d_hitTuple_apc_, quality_d, minHitsPerNtuplet, i);
            }
          });
    }

    if (m_params.doStats_)
      Kokkos::parallel_for(
          "kernel_mark_used",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < d_nCells_()) {
              kernel_mark_used(hhp, d_theCells_, i);
            }
          });

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    cms::kokkos::finalizeBulk<HitContainer, KokkosExecSpace>(d_hitTuple_apc_, tuples_d, execSpace);

    // remove duplicates (tracks that share a doublet)
    Kokkos::parallel_for(
        "kernel_earlyDuplicateRemover",
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < d_nCells_()) {
            kernel_earlyDuplicateRemover(d_theCells_, tuples_d, quality_d, i);
          }
        });

    Kokkos::parallel_for(
        "kernel_countMultiplicity",
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxTuples()),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < tuples_d->nbins()) {
            kernel_countMultiplicity(tuples_d, quality_d, d_tupleMultiplicity_.data(), i);
          }
        });

    cms::kokkos::launchFinalize(d_tupleMultiplicity_, execSpace);

    Kokkos::parallel_for(
        "kernel_fillMultiplicity",
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxTuples()),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < tuples_d->nbins()) {
            kernel_fillMultiplicity(tuples_d, quality_d, d_tupleMultiplicity_.data(), i);
          }
        });

    if (nhits > 1 && m_params.lateFishbone_) {
      int teamSize = 128;
      int stride = 16;
      int blockSize = teamSize / stride;
      int leagueSize = (nhits + blockSize - 1) / blockSize;
#ifdef KOKKOS_BACKEND_SERIAL
      Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, 1};
      // unit stride loop for serial execution
      stride = 1;
#else
      Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, teamSize};
#endif
      Kokkos::parallel_for(
          "latefishbone", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            gpuPixelDoublets::fishbone(
                hhp, d_theCells_, d_nCells_, d_isOuterHitOfCell_, nhits, true, stride, teamMember);
          });
    }
    if (m_params.doStats_) {
      teamSize = 128;
      leagueSize = (std::max(nhits, m_params.maxNumberOfDoublets_) + teamSize - 1) / teamSize;
#ifdef KOKKOS_BACKEND_SERIAL
      policy = Kokkos::TeamPolicy<KokkosExecSpace>(execSpace, leagueSize, 1);
#else
      policy = Kokkos::TeamPolicy<KokkosExecSpace>(execSpace, leagueSize, teamSize);
#endif
      Kokkos::parallel_for(
          "kernel_checkOverflows",
          policy,
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            kernel_checkOverflows(tuples_d,
                                  d_tupleMultiplicity_,
                                  d_hitTuple_apc_,
                                  d_theCells_,
                                  d_nCells_,
                                  d_theCellNeighbors_,
                                  d_theCellTracks_,
                                  d_isOuterHitOfCell_,
                                  nhits,
                                  m_params.maxNumberOfDoublets_,
                                  counters_,
                                  teamMember);
          });
    }
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, KokkosExecSpace const &execSpace) {
    auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    // in principle we can use "nhits" to heuristically dimension the workspace...
    device_isOuterHitOfCell_ =
        Kokkos::View<GPUCACell::OuterHitOfCell *, KokkosExecSpace>("device_isOuterHitOfCell_", std::max(1U, nhits));

    {
      auto isOuterHitOfCell = device_isOuterHitOfCell_;
      Kokkos::parallel_for(
          "initDoublets", Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, nhits), KOKKOS_LAMBDA(const size_t i) {
            assert(isOuterHitOfCell.data());
            isOuterHitOfCell(i).reset();
          });
    }

    device_theCells_ = Kokkos::View<GPUCACell *, KokkosExecSpace>("device_theCells_", m_params.maxNumberOfDoublets_);

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    if (0 == nhits)
      return;  // protect against empty events

    // FIXME avoid magic numbers
    auto nActualPairs = gpuPixelDoublets::nPairs;
    if (!m_params.includeJumpingForwardDoublets_)
      nActualPairs = 15;
    if (m_params.minHitsPerNtuplet_ > 3) {
      nActualPairs = 13;
    }

    assert(nActualPairs <= gpuPixelDoublets::nPairs);
#ifdef KOKKOS_BACKEND_SERIAL
    int stride = 1;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, 1, Kokkos::AUTO()};
#else
    int stride = 4;
    int teamSize = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    int leagueSize = (4 * nhits + teamSize - 1) / teamSize;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, leagueSize, teamSize * stride};
#endif
    // TODO: I do not understand why +2 is needed, the code allocates
    // one uint32_t in addition of the
    // CAConstants::maxNumberOfLayerPairs()
    tempPolicy.set_scratch_size(0, Kokkos::PerTeam((CAConstants::maxNumberOfLayerPairs() + 2) * sizeof(uint32_t)));
    const auto *hhp = hh.view();

    gpuPixelDoublets::getDoubletsFromHisto getdoublets(device_theCells_,
                                                       device_nCells_,
                                                       device_theCellNeighbors_,
                                                       device_theCellTracks_,
                                                       hhp,
                                                       device_isOuterHitOfCell_,
                                                       nActualPairs,
                                                       m_params.idealConditions_,
                                                       m_params.doClusterCut_,
                                                       m_params.doZ0Cut_,
                                                       m_params.doPtCut_,
                                                       m_params.maxNumberOfDoublets_,
                                                       stride);
    Kokkos::parallel_for("getDoubletsFromHisto", tempPolicy, getdoublets);

#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh,
                                                    Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                    KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto const *tuples_d = &tracks_d().hitIndices;
    auto *quality_d = (Quality *)(&tracks_d().m_quality);

    {
      auto const cuts = m_params.cuts_;
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) { kernel_classifyTracks(tuples_d, tracks_d.data(), cuts, quality_d, i); });
    }

    auto theCells = device_theCells_;
    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < device_nCells_()) {
              kernel_fishboneCleaner(theCells.data(), quality_d, i);
            }
          });
    }

    // remove duplicates (tracks that share a doublet)
    {
      auto nCells = device_nCells_;
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < nCells()) {
              kernel_fastDuplicateRemover(theCells.data(), tuples_d, tracks_d.data(), i);
            }
          });
    }

    auto hitToTuple = device_hitToTuple_;
    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_countHitInTracks(tuples_d, quality_d, hitToTuple.data(), i);
            }
          });
      cms::kokkos::launchFinalize(device_hitToTuple_, execSpace);
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_fillHitInTracks(tuples_d, quality_d, hitToTuple.data(), i);
            }
          });
    }

    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      auto hh_view = hh.view();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()), KOKKOS_LAMBDA(const size_t i) {
            if (i < hitToTuple().nbins()) {
              kernel_tripletCleaner(hh_view, tuples_d, tracks_d.data(), quality_d, hitToTuple.data(), i);
            }
          });
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()), KOKKOS_LAMBDA(const size_t i) {
            if (i < hitToTuple().nbins()) {
              kernel_doStatsForHitInTracks(hitToTuple.data(), counters_, i);
            }
          });
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_doStatsForTracks(tuples_d, quality_d, counters_, i);
            }
          });
    }

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    ++iev;
    auto hh_view = hh.view();
    auto const maxPrint = 100;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < std::min(maxPrint, tuples_d->nbins() =)) {
            kernel_print_found_ntuplets(hh_view, tuples_d, tracks_d, quality_d, hitToTuple.data(), 100, iev, i);
          }
        });
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(Kokkos::View<Counters const, KokkosExecSpace> counters) {
#ifdef TODO
    kernel_printCounters<<<1, 1>>>(counters);
#endif
  }

  void CAHitNtupletGeneratorKernels::allocateOnGPU(KokkosExecSpace const &execSpace) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    device_hitToTuple_ = Kokkos::View<HitToTuple, KokkosExecSpace>("device_hitToTuple_");

    device_tupleMultiplicity_ = Kokkos::View<TupleMultiplicity, KokkosExecSpace>("device_tupleMultiplicity_");

    device_hitTuple_apc_ = Kokkos::View<AtomicPairCounter, KokkosExecSpace>("device_hitTuple_apc_");
    device_hitToTuple_apc_ = Kokkos::View<AtomicPairCounter, KokkosExecSpace>("device_hitToTuple_apc_");
    device_nCells_ = Kokkos::View<uint32_t, KokkosExecSpace>("device_nCells_");
    device_tmws_ = Kokkos::View<uint8_t *, KokkosExecSpace>(
        "device_tmws_", std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()));

    Kokkos::deep_copy(execSpace, device_nCells_, 0);

    cms::kokkos::launchZero(device_tupleMultiplicity_, execSpace);
    cms::kokkos::launchZero(device_hitToTuple_, execSpace);  // we may wish to keep it in the edm...
  }

}  // namespace KOKKOS_NAMESPACE
