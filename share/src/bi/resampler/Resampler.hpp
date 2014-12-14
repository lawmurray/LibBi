/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_RESAMPLER_HPP
#define BI_RESAMPLER_RESAMPLER_HPP

#include "../state/State.hpp"
#include "../state/ScheduleElement.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"
#include "../misc/location.hpp"
#include "../traits/resampler_traits.hpp"

namespace bi {
/**
 * Precomputed results for Resampler.
 */
template<Location L>
struct ResamplerPrecompute {
  //
};

/**
 * %Resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * @tparam R Base resampler type.
 */
template<class R>
class Resampler: public R {
public:
  /**
   * Constructor.
   *
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   */
  Resampler(const double essRel = 0.5);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * Get ESS threshold.
   */
  double getEssRel() const;

  /**
   * Set ESS threshold.
   */
  void setEssRel(const double essRel);

  /**
   * Get maximum log-weight.
   */
  double getMaxLogWeight() const;

  /**
   * Set maximum log-weight.
   */
  void setMaxLogWeight(const double maxLogWeight);

  /**
   * Resample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void resample(Random& rng, const ScheduleElement now, S1& s)
      throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Is resampling criterion triggered?
   *
   * @tparam S1 State type.
   *
   * @param now Current step in time schedule.
   * @param s[in,out] State.
   *
   * @return True if resampling is triggered, false otherwise.
   */
  template<class S1>
  bool isTriggered(const ScheduleElement now, S1& s) const
      throw (ParticleFilterDegeneratedException);
  //@}

protected:
  /**
   * Relative ESS threshold.
   */
  double essRel;

  /**
   * Maximum log-weight.
   */
  double maxLogWeight;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include "boost/mpl/if.hpp"

template<class R>
inline bi::Resampler<R>::Resampler(const double essRel) :
    essRel(essRel), maxLogWeight(0.0) {
  /* pre-condition */
  BI_ASSERT(essRel >= 0.0 && essRel <= 1.0);

  //
}

template<class R>
inline double bi::Resampler<R>::getMaxLogWeight() const {
  return maxLogWeight;
}

template<class R>
void bi::Resampler<R>::setMaxLogWeight(const double maxLogWeight) {
  this->maxLogWeight = maxLogWeight;
}

template<class R>
template<class S1>
void bi::Resampler<R>::resample(Random& rng, const ScheduleElement now, S1& s)
    throw (ParticleFilterDegeneratedException) {
  if (isTriggered(now, s)) {
    typename precompute_type<R,S1::location>::type pre;
    typename S1::temp_int_vector_type as1(s.size());

    R::precompute(s.logWeights(), pre);
    R::ancestorsPermute(rng, s.logWeights(), as1, pre);

    s.gather(now, as1);
    set_elements(s.logWeights(), s.logLikelihood);
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
}

template<class R>
template<class S1>
bool bi::Resampler<R>::isTriggered(const ScheduleElement now, S1& s) const
    throw (ParticleFilterDegeneratedException) {
  bool r = false;
  double lW;
  if (now.isObserved() || now.hasBridge()) {
    s.ess = ess_reduce(s.logWeights(), &lW);
    s.logIncrement = lW - s.logLikelihood;
    s.logLikelihood = lW;
    r = essRel >= 1.0 || (essRel > 0.0 && s.ess < essRel * s.size());
  }
  return r;
}

#endif
