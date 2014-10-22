/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Resampler.hpp"

bi::Resampler::Resampler(const double essRel, const double bridgeEssRel) :
    essRel(essRel), bridgeEssRel(bridgeEssRel), maxLogWeight(0.0) {
  /* pre-condition */
  BI_ASSERT(essRel >= 0.0 && essRel <= 1.0);
  BI_ASSERT(bridgeEssRel >= 0.0 && bridgeEssRel <= 1.0);

  //
}

void bi::Resampler::setMaxLogWeight(const double maxLogWeight) {
  this->maxLogWeight = maxLogWeight;
}
