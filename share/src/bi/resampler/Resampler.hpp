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
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  bool isTriggered(const V1 lws) const;

  /**
   * Is ESS-based condition for bridge resampling triggered?
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  bool isTriggeredBridge(const V1 lws) const;

  /**
   * Resample state.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   * @tparam O1 Compatible copy() type.
   *
   * @param rng Random number generator.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * The weights @p lws are set to be uniform after the resampling.
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, V1& lws, V2& as, O1& s)
      throw (ParticleFilterDegeneratedException);
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
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param as Ancestry.
   * @param[in,out] s State.
   */
  template<class V1, class B, Location L>
  static void copy(const V1 as, State<B,L>& s);

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

  /**
   * Normalise log-weights after resampling.
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   *
   * The normalisation is such that the sum of the weights (i.e. @c exp of
   * the components of the vector) is equal to the number of particles.
   */
  template<class V1>
  static void normalise(V1 lws);

  /**
   * Compute effective sample size (ESS) of log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @tparam lws Log-weights.
   *
   * @return ESS.
   */
  template<class V1>
  static typename V1::value_type ess(const V1 lws);

  /**
   * Compute sum of squared errors of ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   *
   * @param lws Log-weights.
   * @param os Offspring.
   *
   * @return Sum of squared errors.
   *
   * This computes the sum of squared errors in the resampling, as in
   * @ref Kitagawa1996 "Kitagawa (1996)":
   *
   * \f[
   * \xi = \sum_{i=1}^P \left(\frac{o_i}{P} - \frac{w_i}{W}\right)^2\,,
   * \f]
   *
   * where \f$W\f$ is the sum of weights.
   */
  template<class V1, class V2>
  static typename V1::value_type sse(const V1 lws, const V2 os);

  /**
   * Compute sum of errors of ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   *
   * @param lws Log-weights.
   * @param os Offspring.
   *
   * @return Sum of errors.
   *
   * This computes the sum of errors in the resampling:
   *
   * \f[
   * \xi = \sum_{i=1}^P \left(\frac{o_i}{P} - \frac{w_i}{W}\right)\,,
   * \f]
   *
   * where \f$W\f$ is the sum of weights.
   */
  template<class V1, class V2>
  static typename V1::value_type se(const V1 lws, const V2 os);
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

template<class V1>
inline bool bi::Resampler::isTriggered(const V1 lws) const {
  return essRel >= 1.0 || ess(lws) < essRel * lws.size();
}

template<class V1>
inline bool bi::Resampler::isTriggeredBridge(const V1 lws) const {
  return bridgeEssRel >= 1.0 || ess(lws) < bridgeEssRel * lws.size();
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

template<class V1, class B, bi::Location L>
void bi::Resampler::copy(const V1 as, State<B,L>& s) {
  s.setRange(s.start(), bi::max(s.size(), as.size()));
  copy(as, s.getDyn());
  s.setRange(s.start(), as.size());
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
      v[i]->resize(v[a]->size(), false);
      *v[i] = *v[a];
    }
  }
}

template<class V1, class M1, class M2>
void bi::Resampler::copy(const V1 as, const M1 X1, M2 X2) {
  gather_rows(as, X1, X2);
}

template<class V1>
void bi::Resampler::normalise(V1 lws) {
  typedef typename V1::value_type T1;
  T1 lW = logsumexp_reduce(lws);
  addscal_elements(lws, bi::log(static_cast<T1>(lws.size())) - lW, lws);
}

template<class V1>
typename V1::value_type bi::Resampler::ess(const V1 lws) {
  typename V1::value_type result = ess_reduce(lws);

  if (result > 0.0) {
    return result;
  } else {
    return 0.0;  // may be nan
  }
}

#endif
