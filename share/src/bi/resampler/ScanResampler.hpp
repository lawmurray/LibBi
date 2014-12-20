/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_SCANRESAMPLER_HPP
#define BI_RESAMPLER_SCANRESAMPLER_HPP

namespace bi {
/**
 * Precomputed results for ScanResampler.
 */
template<Location L>
struct ScanResamplerPrecompute {
  typename loc_temp_vector<L,real>::type Ws;
  real W;
};

/**
 * Base class for resamplers that require an inclusive prefix sum of weights.
 *
 * @ingroup method_resampler
 */
class ScanResampler {
public:
  /**
   * @copydoc Resampler::precompute
   */
  template<class V1, Location L>
  void precompute(const V1 lws, ScanResamplerPrecompute<L>& pre);
};
}

template<class V1, bi::Location L>
void bi::ScanResampler::precompute(const V1 lws,
    ScanResamplerPrecompute<L>& pre) {
  pre.Ws.resize(lws.size(), false);
  sumexpu_inclusive_scan(lws, pre.Ws);
  pre.W = *(pre.Ws.end() - 1);  // sum of weights
}

#endif
