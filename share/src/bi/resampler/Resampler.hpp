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
 * @internal
 *
 * Functor for determining sum of squared errors in particular resampling.
 */
template<class T>
struct resample_se: public std::binary_function<T,int,T> {
  const T lW;
  const T P;
  // ^ oddly, casting o or P in operator()() causes a hang with CUDA 3.1 on
  //   Fermi, so we set the type of P to T instead of int

  /**
   * Constructor.
   */
  CUDA_FUNC_HOST
  resample_se(const T lW, const int P) :
      lW(lW), P(P) {
    //
  }

  /**
   * Apply functor.
   *
   * @param lw Log-weight for this index.
   * @param o Number of offspring for this index.
   *
   * @return Contribution to error for this index.
   */
  CUDA_FUNC_BOTH
  T operator()(const T& lw, const int& o) {
    T eps;

    if (bi::is_finite(lw)) {
      eps = bi::exp(lw - lW) - o / P;  // P of type T, not int, see note above
      eps *= eps;
    } else {
      eps = 0.0;
    }

    return eps;
  }
};

/**
 * @internal
 *
 * Functor for determining sum of errors in particular resampling.
 */
template<class T>
struct resample_e: public std::binary_function<T,int,T> {
  const T lW;
  const T P;
  // ^ oddly, casting o or P in operator()() causes a hang with CUDA 3.1 on
  //   Fermi, so we set the type of P to T instead of int

  /**
   * Constructor.
   */
  CUDA_FUNC_HOST
  resample_e(const T lW, const int P) :
      lW(lW), P(P) {
    //
  }

  /**
   * Apply functor.
   *
   * @param lw Log-weight for this index.
   * @param o Number of offspring for this index.
   *
   * @return Contribution to error for this index.
   */
  CUDA_FUNC_BOTH
  T operator()(const T& lw, const int& o) {
    T eps;

    if (bi::is_finite(lw)) {
      eps = bi::exp(lw - lW) - o / P;  // P of type T, not int, see note above
    } else {
      eps = 0.0;
    }

    return eps;
  }
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
   * Get relative ESS threshold.
   */
  double getEssRel() const;

  /**
   * Set relative ESS threshold.
   */
  void setEssRel(const double essRel = 0.5);

  /**
   * Get maximum log-weight.
   */
  double getMaxLogWeight() const;

  /**
   * Set maximum log-weight.
   */
  void setMaxLogWeight(const double maxLogWeight);

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
   * Correct weights after resampling with proposal.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Vector type.
   * @tparam V2 Vector type.
   *
   * @param as Ancestry.
   * @param qlws Proposal log-weights.
   * @param[in,out] lws Log-weights.
   *
   * Assuming that a resample has been performed using the weights @p qlws,
   * The weights @p lws are set as importance weights, such that if
   * \f$a^i = p\f$, \f$w^i = w^p/w^{*p}\f$, where \f$w^{*p}\f$ are the
   * proposal weights (@p qlws) and \f$w^p\f$ the particle weights (@p lws).
   */
  template<class V1, class V2, class V3>
  static void correct(const V1 as, const V2 qlws, V3 lws);

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
   * but not both. Use permute() to ensure that an ancestry satisfies this
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
   * Is ESS-based condition triggered?
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  bool isTriggered(const V1 lws) const;

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

#include "thrust/inner_product.h"

#include "boost/mpl/if.hpp"

inline double bi::Resampler::getEssRel() const {
  return essRel;
}

inline double bi::Resampler::getMaxLogWeight() const {
  return maxLogWeight;
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

template<class V1, class V2, class V3>
void bi::Resampler::correct(const V1 as, const V2 qlws, V3 lws) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  typedef typename sim_temp_vector<V3>::type vector_type;
  typedef typename V3::value_type T3;

  const int P = as.size();

  vector_type lws1(lws.size());
  lws1 = lws;
  lws.resize(P);

  BOOST_AUTO(iter1,
      thrust::make_permutation_iterator(lws1.begin(), as.begin()));
  BOOST_AUTO(iter2,
      thrust::make_permutation_iterator(qlws.begin(), as.begin()));
  thrust::transform(iter1, iter1 + P, iter2, lws.begin(),
      thrust::minus<T3>());
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
bool bi::Resampler::isTriggered(const V1 lws) const {
  return essRel >= 1.0 || ess(lws) < essRel * lws.size();
}

template<class V1>
typename V1::value_type bi::Resampler::ess(const V1 lws) {
  typename V1::value_type result = ess_reduce(lws);

  if (result > 0.0) {
    return result;
  } else {
    return 0.0; // may be nan
  }
}

template<class V1, class V2>
typename V1::value_type bi::Resampler::sse(const V1 lws, const V2 os) {
  typedef typename V1::value_type T1;

  T1 lW = logsumexp_reduce(lws);

  return thrust::inner_product(lws.begin(), lws.end(), os.begin(),
      T1(0.0), thrust::plus<T1>(), resample_se<T1>(lW, lws.size()));
}

template<class V1, class V2>
typename V1::value_type bi::Resampler::se(const V1 lws, const V2 os) {
  typedef typename V1::value_type T1;

  T1 lW = logsumexp_reduce(lws);

  return thrust::inner_product(lws.begin(), lws.end(), os.begin(),
      T1(0.0), thrust::plus<T1>(), resample_e<T1>(lW, lws.size()));
}

#endif
