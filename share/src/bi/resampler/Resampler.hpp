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
#include "../random/Random.hpp"
#include "../misc/exception.hpp"
#include "../misc/location.hpp"
#include "../traits/resampler_traits.hpp"

namespace bi {
/**
 * %Resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class Resampler {
public:
  /**
   * Constructor.
   *
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   */
  Resampler(const double essRel = 0.5, const double essRelBridge = 0.5);

  /**
   * @name High-level interface.
   */
  //@{
  /**
   * Get maximum log-weight.
   */
  double getMaxLogWeight() const;

  /**
   * Set maximum log-weight.
   */
  void setMaxLogWeight(const double maxLogWeight);

  /**
   * Is ESS-based condition triggered?
   *
   * @tparam S1 State type.
   *
   * @param now Current step in time schedule.
   * @param s State.
   * @param[out] lW If given, the log of the sum of weights is written to this
   * variable.
   *
   * @return True if resampling is triggered, false otherwise.
   */
  template<class S1>
  bool isTriggered(const ScheduleElement now, const S1& s, double* lW) const;
  //@}

  /**
   * @name Low-level interface.
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * Select offspring.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] os Offspring.
   * @param P Total number of offspring to select.
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);

  /**
   * Select cumulative offspring.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] Os Cimulative offspring.
   * @param P Total number of offspring to select.
   */
  template<class V1, class V2>
  void cumulativeOffspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);

  /**
   * Compute offspring vector from ancestors vector.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Integral vector type.
   *
   * @param as Ancestors.
   * @param[out] os Offspring.
   */
  template<class V1, class V2>
  static void ancestorsToOffspring(const V1 as, V2 os);

  /**
   * Compute ancestor vector from offspring vector.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Integral vector type.
   *
   * @param os Offspring.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  static void offspringToAncestors(const V1 os, V2 as);

  /**
   * Compute already-permuted ancestor vector from offspring vector.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Integral vector type.
   *
   * @param os Offspring.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  static void offspringToAncestorsPermute(const V1 os, V2 as);

  /**
   * Compute ancestor vector from cumulative offspring vector.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Integral vector type.
   *
   * @param Os Cumulative offspring.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestors(const V1 Os, V2 as);

  /**
   * Compute already-permuted ancestor vector from cumulative offspring
   * vector.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Integral vector type.
   *
   * @param Os Cumulative offspring.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestorsPermute(const V1 Os, V2 as);

  /**
   * Permute ancestors to permit in-place copy.
   *
   * @tparam V1 Integral vector type.
   *
   * @param[in,out] as Ancestry.
   */
  template<class V1>
  static void permute(V1 as);

  /**
   * In-place copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param as Ancestry.
   * @param[in,out] X Matrix. Rows of the matrix are copied.
   *
   * The copy is performed in-place. For each particle @c i that is to be
   * preserved (i.e. its offspring count is at least 1), @c a[i] should equal
   * @c i. This ensures that all particles are either read or (over)written,
   * constraint.
   */
  template<class V1, class M1>
  static void copy(const V1 as, M1 X);

  /**
   * In-place copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam T1 Assignable type.
   *
   * @param as Ancestry.
   * @oaram[in,out] v STL vector.
   */
  template<class V1, class T1>
  static void copy(const V1 as, std::vector<T1*>& v);

  /**
   * Copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   *
   * @param X1 Input matrix.
   * @param as Ancestry.
   * @param X2 Output matrix.
   */
  template<class V1, class M1, class M2>
  static void copy(const V1 as, const M1 X1, M2 X2);
  //@}

protected:
  /**
   * Relative ESS threshold.
   */
  double essRel;

  /**
   * Realtive ESS threshold for bridge sampling.
   */
  double bridgeEssRel;

  /**
   * Maximum log-weight.
   */
  double maxLogWeight;
};

/**
 * Resampler implementation on host.
 */
class ResamplerHost {
public:
  /**
   * @copydoc Resampler::ancestorsToOffspring()
   */
  template<class V1, class V2>
  static void ancestorsToOffspring(const V1 as, V2 os);

  /**
   * @copydoc Resampler::offspringToAncestors()
   */
  template<class V1, class V2>
  static void offspringToAncestors(const V1 os, V2 as);

  /**
   * @copydoc Resampler::offspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void offspringToAncestorsPermute(const V1 os, V2 as);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestors()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestors(const V1 Os, V2 as);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestorsPermute(const V1 Os, V2 as);

  /**
   * @copydoc Resampler::permute()
   */
  template<class V1>
  static void permute(V1 as);
};

/**
 * Resampler implementation on device.
 */
class ResamplerGPU {
public:
  /**
   * @copydoc Resampler::ancestorsToOffspring()
   */
  template<class V1, class V2>
  static void ancestorsToOffspring(const V1 as, V2 os);

  /**
   * @copydoc Resampler::offspringToAncestors()
   */
  template<class V1, class V2>
  static void offspringToAncestors(const V1 os, V2 as);

  /**
   * @copydoc Resampler::offspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void offspringToAncestorsPermute(const V1 os, V2 as);

  /**
   * Like offspringToAncestorsPermute(), but only performs first stage of
   * permutation. Second stage should be completed with postPermute().
   */
  template<class V1, class V2, class V3>
  static void offspringToAncestorsPrePermute(const V1 os, V2 as, V3 is);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestors()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestors(const V1 Os, V2 as);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestorsPermute(const V1 Os, V2 as);

  /**
   * Like cumulativeOffspringToAncestorsPermute(), but only performs first
   * stage of permutation. Second stage should be completed with
   * postPermute().
   */
  template<class V1, class V2, class V3>
  static void cumulativeOffspringToAncestorsPrePermute(const V1 Os, V2 as,
      V3 is);

  /**
   * @copydoc Resampler::permute()
   */
  template<class V1>
  static void permute(V1 as);

  /**
   * First stage of permutation.
   *
   * @tparam V1 Integer vector type.
   * @tparam V2 Integer vector type.
   *
   * @param as Input ancestry.
   * @param is[out] Claims.
   */
  template<class V1, class V2>
  static void prePermute(const V1 as, V2 is);

  /**
   * Second stage of permutation.
   *
   * @tparam V1 Integer vector type.
   * @tparam V2 Integer vector type.
   * @tparam V3 Integer vector type.
   *
   * @param as Input ancestry.
   * @param is Claims, as output from pre-permute function.
   * @param[out] cs Output, permuted ancestry.
   */
  template<class V1, class V2, class V3>
  static void postPermute(const V1 as, const V2 is, V3 cs);
};
}

#include "../host/resampler/ResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/ResamplerGPU.cuh"
#endif

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include "boost/mpl/if.hpp"

inline double bi::Resampler::getMaxLogWeight() const {
  return maxLogWeight;
}

template<class S1>
bool bi::Resampler::isTriggered(const ScheduleElement now, const S1& s,
    double* lW) const {
  const int P = s.size();
  double ess = ess_reduce(s.logWeights(), lW);
  return (now.isObserved() && (essRel >= 1.0 || ess < essRel * P))
      || (now.isBridge() && (bridgeEssRel >= 1.0 || ess < bridgeEssRel * P));
}

template<class V1, class V2>
void bi::Resampler::ancestorsToOffspring(const V1 as, V2 os) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::ancestorsToOffspring(as, os);
}

template<class V1, class V2>
void bi::Resampler::offspringToAncestors(const V1 os, V2 as) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::offspringToAncestors(os, as);
}

template<class V1, class V2>
void bi::Resampler::offspringToAncestorsPermute(const V1 os, V2 as) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::offspringToAncestorsPermute(os, as);
}

template<class V1, class V2>
void bi::Resampler::cumulativeOffspringToAncestors(const V1 Os, V2 as) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2>
void bi::Resampler::cumulativeOffspringToAncestorsPermute(const V1 Os,
    V2 as) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::cumulativeOffspringToAncestorsPermute(Os, as);
}

template<class V1>
void bi::Resampler::permute(const V1 as) {
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
  impl::permute(as);
}

template<class V1, class M1>
void bi::Resampler::copy(const V1 as, M1 s) {
  gather_rows(as, s, s);
}

template<class V1, class T1>
void bi::Resampler::copy(const V1 as, std::vector<T1*>& v) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);

  // don't use OpenMP for this, causing segfault with Intel compiler, and
  // with CUDA, possibly due to different CUDA contexts with different
  // threads playing with the resize and assignment
  for (int i = 0; i < as.size(); ++i) {
    int a = as(i);
    if (i != a) {
      *v[i] = *v[a];
    }
  }
}

template<class V1, class M1, class M2>
void bi::Resampler::copy(const V1 as, const M1 X1, M2 X2) {
  gather_rows(as, X1, X2);
}

#endif
