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
 */
class Resampler {
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
   * Get maximum log-weight.
   */
  double getMaxLogWeight() const;

  /**
   * Set maximum log-weight.
   */
  void setMaxLogWeight(const double maxLogWeight);

  /**
   * Is resampling criterion triggered?
   *
   * @tparam V1 Vector type.
   *
   * @param now Current step in time schedule.
   * @param lws Log-weights.
   * @param[out] lW If given, the log of the sum of weights is written to this
   * variable.
   * @param[out] ess If given, the ESS of weihts is written to this variable.
   *
   * @return True if resampling is triggered, false otherwise.
   */
  template<class V1>
  bool isTriggered(const ScheduleElement now, const V1 lws, double* lW = NULL,
      double* ess = NULL) const throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Perform precomputations.
   *
   * @tparam V1 Vector type.
   * @tparam pre Precompute type.
   *
   * @param lws Log-weights.
   * @param pre Precompute object.
   */
  template<class V1, Location L>
  void precompute(const V1 lws, ResamplerPrecompute<L>& pre);

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
   * Maximum log-weight.
   */
  double maxLogWeight;
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
bool bi::Resampler::isTriggered(const ScheduleElement now, const V1 lws,
    double* lW, double* ess) const throw (ParticleFilterDegeneratedException) {
  double P = lws.size();
  bool r = false;

  if (now.isObserved() || now.hasBridge()) {
    if (essRel >= 1.0) {
      r = true;
      if (ess != NULL) {
        *ess = ess_reduce(lws, lW);  // computes lW as well if not NULL
      } else if (lW != NULL) {
        *lW = logsumexp_reduce(lws) - bi::log(P);  // computes lW only
      }
    } else {
      double ess1 = ess_reduce(lws, lW);  // computes lW as well if not NULL
      r = ess1 < essRel * P;
      if (ess != NULL) {
        *ess = ess1;
      }
    }
  }
  return r;
}

template<class V1, bi::Location L>
void bi::Resampler::precompute(const V1 lws, ResamplerPrecompute<L>& pre) {
  //
}

template<class V1, class V2>
void bi::Resampler::ancestorsToOffspring(const V1 as, V2 os) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::ancestorsToOffspring(as, os);
}

template<class V1, class V2>
void bi::Resampler::offspringToAncestors(const V1 os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::offspringToAncestors(os, as);
}

template<class V1, class V2>
void bi::Resampler::offspringToAncestorsPermute(const V1 os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::offspringToAncestorsPermute(os, as);
}

template<class V1, class V2>
void bi::Resampler::cumulativeOffspringToAncestors(const V1 Os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2>
void bi::Resampler::cumulativeOffspringToAncestorsPermute(const V1 Os,
    V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::cumulativeOffspringToAncestorsPermute(Os, as);
}

template<class V1>
void bi::Resampler::permute(const V1 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
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
