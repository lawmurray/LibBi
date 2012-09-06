/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MISC_HPP
#define BI_METHOD_MISC_HPP

#include "../math/scalar.hpp"
#include "../cache/Cache2D.hpp"

#include <set>

namespace bi {
/**
 * Optimisation mode.
 */
enum OptimiserMode {
  /**
   * Maximum likelihood estimation.
   */
  MAXIMUM_LIKELIHOOD,

  /**
   * Maximum a posteriori.
   */
  MAXIMUM_A_POSTERIORI
};

/**
 * Compute next time for given delta that is greater than the current time.
 *
 * @ingroup method
 *
 * @param t Current time.
 * @param delta Time step.
 *
 * @return If @p delta is positive, next time that is a multiple of @p delta.
 * If @p delta is negative, previous time that is a multiple of
 * <tt>abs(delta)</tt>.
 */
real gt_step(const real t, const real delta);

/**
 * Compute next time for given delta that is greater than or equal to the
 * current time.
 *
 * @ingroup method
 *
 * @param t Current time.
 * @param delta Time step.
 *
 * @return If @p t a multiple of @p delta, then @p t. If @p delta is positive,
 * next time that is a multiple of @p delta. If @p delta is negative,
 * previous time that is a multiple of <tt>abs(delta)</tt>.
 */
real ge_step(const real t, const real delta);

/**
 * Number of time steps in closed interval.
 *
 * @ingroup method
 *
 * @param t Time.
 * @param delta Time step (positive).
 *
 * @return Number of multiples of @p delta on the interval <tt>[0,t]</tt>.
 */
int le_steps(const real t, const real delta);

/**
 * Number of time steps in open interval.
 *
 * @ingroup method
 *
 * @param t Time.
 * @param delta Time step (positive).
 *
 * @return Number of multiples of @p delta on the interval <tt>[0,t)</tt>.
 */
int lt_steps(const real t, const real delta);

/**
 * Number of time steps in closed interval
 *
 * @ingroup method
 *
 * @param ti Start of interval.
 * @param tj End of interval.
 * @param delta Time step.
 *
 * @return If @p delta is positive, number of multiples of @p delta on the
 * interval <tt>[ti,tj]</tt>. If @p delta is negative, number of multiples of
 * <tt>abs(delta)</tt> on the interval <tt>[tj,ti]</tt>;
 */
int le_steps(const real ti, const real tj, const real delta);

/**
 * Number of time steps in open interval
 *
 * @ingroup method
 *
 * @param ti Start of interval.
 * @param tj End of interval.
 * @param delta Time step.
 *
 * @return If @p delta is positive, number of multiples of @p delta on the
 * interval <tt>[ti,tj)</tt>. If @p delta is negative, number of multiples of
 * <tt>abs(delta)</tt> on the interval <tt>[tj,ti)</tt>;
 */
int lt_steps(const real ti, const real tj, const real delta);

/**
 * Insert elements into set, with offset.
 *
 * @ingroup method
 *
 * @param xs Set into which to insert.
 * @param first Iterator to first element in range of values to be inserted.
 * @param last Iterator to last element in range of values to be inserted.
 * @param offset Offset to add to each value to be inserted.
 */
template<class T, class InputIterator>
void offset_insert(std::set<T>& xs, InputIterator first, InputIterator last,
    const T offset = 0);

/**
 * Summarise particle filter with one weight stage.
 *
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param lws Log-weights cache.
 * @param[out] ll Marginal log-likelihood.
 * @param[out] lls Log-likelihood at each observation.
 * @param[out] ess Effective sample size at each observation.
 *
 * Any number of the parameters may be @c NULL.
 */
template<class T1, class V2, class V3>
void summarise_pf(const Cache2D<T1>& lws, T1* ll, V2* lls, V3* ess);

/**
 * Summarise particle filter with two weight stages.
 *
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param lw1s Stage-1 log-weights cache.
 * @param lw2s Stage-2 log-weights cache.
 * @param[out] ll Marginal log-likelihood.
 * @param[out] lls Log-likelihood at each observation.
 * @param[out] ess Effective sample size at each observation.
 *
 * Any number of the parameters may be @c NULL.
 *
 * Uses the marginal likelihood estimator of @ref Pitt2002 "Pitt (2002)" and
 * @ref Pitt2010 "Pitt et al. (2010)".
 */
template<class T1, class V1, class V2>
void summarise_apf(const Cache2D<T1>& lw1s, const Cache2D<T1>& lw2s, T1* ll,
    V1* lls, V2* ess);

}

#include "../math/function.hpp"
#include "../math/temp_vector.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

inline real bi::gt_step(const real t, const real delta) {
  return delta*bi::ceil((t + 1.0e-3*delta)/delta);
}

inline real bi::ge_step(const real t, const real delta) {
  return delta*bi::ceil((t - 1.0e-3*delta)/delta);
}

inline int bi::le_steps(const real t, const real delta) {
  return static_cast<int>(bi::ceil((t + 1.0e-3*delta)/delta));
}

inline int bi::lt_steps(const real t, const real delta) {
  return static_cast<int>(bi::ceil((t - 1.0e-3*delta)/delta));
}

inline int bi::le_steps(const real ti, const real tj, const real delta) {
  int steps;
  if (tj >= ti) {
    steps = le_steps(tj, delta) - lt_steps(ti, delta);
  } else {
    steps = le_steps(ti, delta) - lt_steps(tj, delta);
  }
  return steps;
}

inline int bi::lt_steps(const real ti, const real tj, const real delta) {
  int steps;
  if (tj >= ti) {
    steps = lt_steps(tj, delta) - lt_steps(ti, delta);
  } else {
    steps = lt_steps(ti, delta) - lt_steps(tj, delta);
  }
  return steps;
}

template<class T, class InputIterator>
void bi::offset_insert(std::set<T>& xs, InputIterator first, InputIterator last,
    const T offset = 0) {
  InputIterator iter;
  for (iter = first; iter != last; ++iter) {
    xs.insert(*iter + offset);
  }
}

template<class T1, class V2, class V3>
void bi::summarise_pf(const Cache2D<T1>& lws, T1* ll, V2* lls, V3* ess) {
  const int T = lws.size();
  typename temp_host_vector<real>::type ess1(T), lls1(T), lls2(T);
  double ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real logsum1, sum1, sum2;
  for (n = 0; n < T; ++n) {
    typename temp_host_vector<real>::type lws1(lws.get(n).size());
    lws1 = lws.get(n);

    bi::sort(lws1);
    logsum1 = logsumexp_reduce(lws1);
    sum1 = bi::exp(logsum1);
    sum2 = sumexpsq_reduce(lws1);

    lls1(n) = logsum1 - bi::log(static_cast<real>(lws1.size()));
    ess1(n) = (sum1*sum1)/sum2;
  }

  /* compute marginal log-likelihood */
  lls2 = lls1;
  bi::sort(lls2);
  ll1 = sum_reduce(lls2);

  /* write to output params, where given */
  if (ll != NULL) {
    *ll = ll1;
  }
  if (lls != NULL) {
    *lls = lls1;
  }
  if (ess != NULL) {
    *ess = ess1;
  }
}

template<class T1, class V1, class V2>
void bi::summarise_apf(const Cache2D<T1>& lw1s, const Cache2D<T1>& lw2s,
    T1* ll, V1* lls, V2* ess) {
  const int T = lw1s.size();
  typename temp_host_vector<real>::type ess1(T), lls1(T), lls2(T);
  real ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real logsum1, logsum2, sum2, sum3;
  for (n = 0; n < T; ++n) {
    typename temp_host_vector<real>::type lw1s1(lw1s.get(n).size());
    lw1s1 = lw1s.get(n);
    typename temp_host_vector<real>::type lw2s1(lw2s.get(n).size());
    lw2s1 = lw2s.get(n);
    BI_ASSERT(lw1s1.size() == lw2s1.size());

    bi::sort(lw1s1);
    bi::sort(lw2s1);

    logsum1 = logsumexp_reduce(lw1s1);
    logsum2 = logsumexp_reduce(lw2s1);
    sum2 = bi::exp(logsum2);
    sum3 = sumexpsq_reduce(lw2s1);

    lls1(n) = logsum1 + logsum2 - 2.0*bi::log(static_cast<real>(lw1s1.size()));
    ess1(n) = (sum2*sum2)/sum3;
  }

  /* compute marginal log-likelihood */
  lls2 = lls1;
  bi::sort(lls2);
  ll1 = sum_reduce(lls2);

  /* write to output params, where given */
  if (ll != NULL) {
    *ll = ll1;
  }
  if (lls != NULL) {
    *lls = lls1;
  }
  if (ess != NULL) {
    *ess = ess1;
  }
}

#endif
