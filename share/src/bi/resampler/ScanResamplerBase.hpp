/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_SCANRESAMPLERBASE_HPP
#define BI_RESAMPLER_SCANRESAMPLERBASE_HPP

#include "ResamplerBase.hpp"

namespace bi {
/**
 * Precomputed results for ScanResamplerBase.
 */
template<Location L>
struct ScanResamplerBasePrecompute {
  typename loc_temp_vector<L,real>::type Ws;
  real W;
};

/**
 * Base class for resamplers that require an inclusive prefix sum of weights.
 *
 * @ingroup method_resampler
 */
class ScanResamplerBase: public ResamplerBase {
public:
  /**
   * @copydoc ResamplerBase::ResamplerBase
   */
  ScanResamplerBase(const double essRel = 0.5,
      const double essRelBridge = 0.5);

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc ResamplerBase::precompute
   */
  template<class V1, Location L>
  void precompute(const V1 lws, ScanResamplerBasePrecompute<L>& pre);
  //@}
};
}

template<class V1, bi::Location L>
void bi::ScanResamplerBase::precompute(const V1 lws,
    ScanResamplerBasePrecompute<L>& pre) {
  pre.Ws.resize(lws.size(), false);
  sumexpu_inclusive_scan(lws, pre.Ws);
  pre.W = *(pre.Ws.end() - 1);  // sum of weights
}

#endif
