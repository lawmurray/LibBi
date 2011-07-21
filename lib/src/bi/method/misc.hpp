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
#include "../buffer/Cache2D.hpp"

namespace bi {
/**
 * Flags to control filter behaviour with regard to p- and s-nodes.
 */
enum StaticHandling {
  /**
   * All trajectories share p- and s-nodes.
   */
  STATIC_SHARED,

  /**
   * Each trajectory has own p- and s-nodes.
   */
  STATIC_OWN
};

/**
 * Behaviour types for handling initial conditions in ParticleMCMC.
 */
enum InitialConditionType {
  /**
   * All trajectories share same initial condition, which becomes part of
   * the MCMC process.
   */
  INITIAL_CONDITIONED,

  /**
   * Each trajectory has own initial condition, which is drawn from the
   * prior as part of the filtering process.
   */
  INITIAL_SAMPLED
};

/**
 * Filter types for likelihood calculation in ParticleMCMC.
 */
enum FilterType {
  /**
   * Unconditioned filter.
   */
  UNCONDITIONED,

  /**
   * Filter conditioned on current state trajectory.
   */
  CONDITIONED
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

#include "../math/temp_vector.hpp"
#include "../math/scalar.hpp"

#include "boost/typeof/typeof.hpp"

#include <cmath>

inline real bi::gt_step(const real t, const real delta) {
  return delta*ceil((t + 1.0e-3*delta)/delta);
}

inline real bi::ge_step(const real t, const real delta) {
  return delta*ceil((t - 1.0e-3*delta)/delta);
}

inline int bi::le_steps(const real t, const real delta) {
  return static_cast<int>(ceil((t + 1.0e-3*delta)/delta));
}

inline int bi::lt_steps(const real t, const real delta) {
  return static_cast<int>(ceil((t - 1.0e-3*delta)/delta));
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
  BOOST_AUTO(ess1, host_temp_vector<real>(T));
  BOOST_AUTO(lls1, host_temp_vector<real>(T));
  BOOST_AUTO(lls2, host_temp_vector<real>(T));
  double ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real logsum1, sum1, sum2;
  for (n = 0; n < T; ++n) {
    BOOST_AUTO(lws1, duplicate_vector(lws.get(n)));

    bi::sort(lws1->begin(), lws1->end());
    logsum1 = log_sum_exp(lws1->begin(), lws1->end(), 0.0);
    sum1 = exp(logsum1);
    sum2 = sum_exp_square(lws1->begin(), lws1->end(), 0.0);

    (*lls1)(n) = logsum1 - std::log(lws1->size());
    (*ess1)(n) = (sum1*sum1)/sum2;

    delete lws1;
  }

  /* compute marginal log-likelihood */
  *lls2 = *lls1;
  bi::sort(lls2->begin(), lls2->end());
  ll1 = bi::sum(lls2->begin(), lls2->end(), 0.0);

  /* write to output params, where given */
  if (ll != NULL) {
    *ll = ll1;
  }
  if (lls != NULL) {
    *lls = *lls1;
  }
  if (ess != NULL) {
    *ess = *ess1;
  }

  delete ess1;
  delete lls1;
  delete lls2;
}

template<class T1, class V1, class V2>
void bi::summarise_apf(const Cache2D<T1>& lw1s, const Cache2D<T1>& lw2s,
    T1* ll, V1* lls, V2* ess) {
  const int T = lw1s.size();
  BOOST_AUTO(ess1, host_temp_vector<real>(T));
  BOOST_AUTO(lls1, host_temp_vector<real>(T));
  BOOST_AUTO(lls2, host_temp_vector<real>(T));
  real ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real logsum1, logsum2, sum2, sum3;
  for (n = 0; n < T; ++n) {
    BOOST_AUTO(lw1s1, duplicate_vector(lw1s.get(n)));
    BOOST_AUTO(lw2s1, duplicate_vector(lw2s.get(n)));
    assert (lw1s1->size() == lw2s1->size());

    bi::sort(lw1s1->begin(), lw1s1->end());
    bi::sort(lw2s1->begin(), lw2s1->end());

    logsum1 = log_sum_exp(lw1s1->begin(), lw1s1->end(), 0.0);
    logsum2 = log_sum_exp(lw2s1->begin(), lw2s1->end(), 0.0);
    sum2 = exp(logsum2);
    sum3 = sum_exp_square(lw2s1->begin(), lw2s1->end(), 0.0);

    (*lls1)(n) = logsum1 + logsum2 - 2*std::log(lw1s1->size());
    (*ess1)(n) = (sum2*sum2)/sum3;

    delete lw1s1;
    delete lw2s1;
  }

  /* compute marginal log-likelihood */
  *lls2 = *lls1;
  bi::sort(lls2->begin(), lls2->end());
  ll1 = bi::sum(lls2->begin(), lls2->end(), 0.0);

  /* write to output params, where given */
  if (ll != NULL) {
    *ll = ll1;
  }
  if (lls != NULL) {
    *lls = *lls1;
  }
  if (ess != NULL) {
    *ess = *ess1;
  }

  delete ess1;
  delete lls1;
  delete lls2;
}

#endif
