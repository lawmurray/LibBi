/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1246 $
 * $Date: 2011-01-31 16:23:46 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP
#define BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP

#include "AuxiliaryParticleFilter.hpp"

/**
 * @todo Replace with variadic private member function.
 */
#define MAKE_ARGS(output, ...) \
    BOOST_AUTO(args_##output, thrust::make_tuple(__VA_ARGS__)); \
    BOOST_AUTO(output, thrust::make_zip_iterator(args_##output));

namespace bi {
/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$A\f$ matrix.
 */
template<class T1>
struct disturbance_compute_A {
  template<class A1>
  CUDA_FUNC_BOTH real operator()(const A1& x) {
    T1 U = thrust::get<0>(x);
    T1 mu = thrust::get<1>(x);
    T1 inv_sigma = thrust::get<2>(x);

    return (U - mu)*inv_sigma;
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$\mathbf{v}\f$ vector.
 */
template<class T1>
struct disturbance_compute_v {
  T1 det;

  CUDA_FUNC_HOST disturbance_compute_v(const T1 det) : det(det) {
    //
  }

  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 h = thrust::get<0>(x);

    return det*CUDA_EXP(h);
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$dH/d\boldsymbol{\mu}\f$ vector.
 */
template<class T1>
struct disturbance_compute_dH1 {
  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 A = thrust::get<0>(x);
    T1 inv_sigma = thrust::get<1>(x);

    return -A*inv_sigma;
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$dH/d\boldsymbol{\sigma}^{-1}\f$ vector.
 */
template<class T1>
struct disturbance_compute_dH2 {
  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 A = thrust::get<0>(x);
    T1 U = thrust::get<1>(x);
    T1 mu = thrust::get<2>(x);
    return A*(U - mu);
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$dV/d\boldsymbol{\mu}\f$ vector.
 */
template<class T1>
struct disturbance_compute_dV1 {
  T1 det;


  CUDA_FUNC_HOST disturbance_compute_dV1(const T1 det) : det(det) {
    //
  }

  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 dH = thrust::get<0>(x);
    T1 h = thrust::get<1>(x);

    return dH*CUDA_EXP(h)*det;
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$dV/d\boldsymbol{\sigma}^{-1}\f$ vector.
 */
template<class T1>
struct disturbance_compute_dV2 {
  T1 det;

  CUDA_FUNC_HOST disturbance_compute_dV2(const T1 det) : det(det) {
    //
  }

  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 dH = thrust::get<0>(x);
    T1 h = thrust::get<1>(x);
    T1 inv_sigma = thrust::get<2>(x);

    return -dH*CUDA_EXP(h)*det/inv_sigma;
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$\mathbf{b}\f$ vector.
 */
template<class T1>
struct disturbance_compute_b {
  T1 lhat;

  CUDA_FUNC_HOST disturbance_compute_b(const T1 lhat) : lhat(lhat) {
    //
  }

  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 v = thrust::get<0>(x);
    T1 w = thrust::get<1>(x);

    return v*w - lhat;
  }
};

/**
 * @internal
 *
 * @tparam T1 Scalar type.
 *
 * Compute \f$\mathbf{b}\f$ vector.
 */
template<class T1>
struct disturbance_compute_dF {
  template<class A1>
  CUDA_FUNC_BOTH T1 operator()(const A1& x) {
    T1 b = thrust::get<0>(x);
    T1 w = thrust::get<1>(x);
    T1 dV = thrust::get<2>(x);

    return static_cast<T1>(2.0)*b*w*dV;
  }
};

/**
 * Disturbance particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::DisturbanceParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class DisturbanceParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  DisturbanceParticleFilter(B& m, Random& rng, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc #concept::Filter::filter()
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL);

  /**
   * @copydoc #concept::Filter::filter(real, const V1&, R*)
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Look ahead to next observation.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R Resampler type.
   *
   * @param[in,out] s State.
   * @param[in,out] lw1s Stage 1 log-weights.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * Looks forward to the next observation, setting r-nodes in @p s to improve
   * support at this next time, and adjusting log-weights @p lws accordingly.
   */
  template<Location L, class V1, class V2, class R>
  void lookahead(State<L>& s, V1& lw1s, V1& lw2s, V2& as, R* resam);
  //@}

private:
  /**
   * Minimise variance in likelihood estimator via gradient descent on
   * proposal mean.
   */
  template<class M1, class V1, class V2>
  void minimise(const M1& U, const V1& lws, V2& mu);

  /**
   * Minimise variance in likelihood estimator via gradient descent on
   * proposal mean and standard deviation.
   */
  template<class M1, class V1>
  void minimise(M1& U, V1& lws);
};

/**
 * Factory for creating DisturbanceParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see DisturbanceParticleFilter
 */
template<Location CL = ON_HOST>
struct DisturbanceParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return DisturbanceParticleFilter object. Caller has ownership.
   *
   * @see DisturbanceParticleFilter::DisturbanceParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>* create(B& m,
      Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs,
        out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::DisturbanceParticleFilter(
    B& m, Random& rng, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs, out) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam) {
  /* pre-condition */
  assert (T > this->state.t);

  int n = 0, r = 0;

  BOOST_AUTO(lw1s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(lw2s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  init(theta, *lw2s, *as);
  while (this->state.t < T) {
    lookahead(s, *lw1s, *lw2s, *as, resam);
    predict(T, s);
    correct(s, *lw2s);
    output(n, s, r, *lw2s, *as);
    ++n;
//    r = true;
    r = this->state.t < T; // no need to resample at last time
    if (r) {
      resample(s, *lw2s, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lw1s;
  delete lw2s;
  delete as;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam) {
  assert(false);
}

/**
 * @todo Currently assumes only one day between observations.
 */
template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::lookahead(State<L>& s,
    V1& lw1s, V1& lw2s, V2& as, R* resam) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;
  typedef typename locatable_temp_matrix<L,real>::type temp_matrix_type;

  const real to = this->oyUpdater.getTime();
  if (to >= this->state.t) {
    const int P = s.size();
    const int ND = this->m.getNetSize(D_NODE);
    const int NC = this->m.getNetSize(C_NODE);
    const int NR = this->m.getNetSize(R_NODE);

    temp_matrix_type X(P, ND + NC);
    temp_vector_type lw1s1(P), lqs(P), lps(P), ws1(P);

    /* store current state */
    columns(X, 0, ND) = s.get(D_NODE);
    columns(X, ND, NC) = s.get(C_NODE);
    this->mark();

    /* auxiliary simulation forward */
    predict(to, s);
    lw1s = lw2s;
    correct(s, lw1s);
    //resam->resample(lw1s, lw2s, as);

    /* disturbance adjustment */
    lw1s1 = lw1s;
    minimise(s.get(R_NODE), lw1s1);

//    element_exp(lw1s1.begin(), lw1s1.end(), ws1.begin());
//    mean(s.get(R_NODE), ws1, qU.mean());
//    var(columns(s.get(R_NODE), 6, 1), ws1, qU.mean(), diagonal(qU.cov()));
//
//    /* sample */
//    qU.samples(this->rng, s.get(R_NODE));
//    this->rUpdater.skipNext();
//    qU.logDensities(s.get(R_NODE), lqs);
//    pU.logDensities(s.get(R_NODE), lps);
//
//    /* correct weights */
//    axpy(1.0, lps, lw2s);
//    axpy(-1.0, lqs, lw2s);

    /* restore previous state */
    //resam->copy(as, X);
    s.get(D_NODE) = columns(X, 0, ND);
    s.get(C_NODE) = columns(X, ND, NC);
    this->restore();

    synchronize();

    /* post-condition */
    assert (this->sim.getTime() == this->getTime());
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class M1, class V1, class V2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::minimise(const M1& U, const V1& lw1s, V2& mu) {
  /* pre-condition */
  assert (U.size1() == lw1s.size());
  assert (U.size2() == mu.size());

  const int NR = U.size2();
  const int P = U.size1();

  BOOST_AUTO(lv, temp_vector<V1>(P));
  BOOST_AUTO(a, temp_vector<V1>(P));
  BOOST_AUTO(b, temp_vector<V1>(P));
  BOOST_AUTO(c, temp_vector<V1>(P));
  BOOST_AUTO(df, temp_vector<V1>(NR));
  BOOST_AUTO(df0, temp_vector<V1>(NR));
  BOOST_AUTO(mu0, temp_vector<V1>(NR));

  int n = 0;
  real delta;
  real f, f0, diff, C;
  real lhat = bi::sum_exp(lw1s.begin(), lw1s.end(), REAL(0.0))/P;

  /* initialise */
  mu.clear();

  /* initial function derivatives */
  gemv(-1.0, U, mu, 0.0, *a);
  axpy(1.0, lw1s, *a);
  element_exp(a->begin(), a->end(), a->begin());
  scal(0.5*dot(mu,mu), *a);

  gdmv(1.0, *a, *a, 0.0, *c);
  axpy(-lhat, *a, *c);

  C = bi::sum(c->begin(), c->end(), REAL(0.0));
  axpy(2.0*C, mu, *df, true);
  gemv(-2.0, U, *c, 1.0, *df, 'T');

  /* initial function value */
  f = dot(*b, *b);

  /* accept */
  f0 = f;
  *df0 = *df;
  *mu0 = mu;
  delta = 1.0/f0;

//  std::cerr << n << ": f=" << f0 << std::endl;
  ++n;

  do {
    /* take step */
    mu = *mu0;
    axpy(-delta, *df0, mu);

    /* function derivatives */
    gemv(-1.0, U, mu, 0.0, *a);
    axpy(1.0, lw1s, *a);
    element_exp(a->begin(), a->end(), a->begin());
    scal(exp(0.5*dot(mu,mu)), *a);

    gdmv(1.0, *a, *a, 0.0, *c);
    axpy(-lhat, *a, *c);

    bi::fill(b->begin(), b->end(), -lhat);
    axpy(1.0, *a, *b);
    C = dot(*a, *b);

    axpy(2.0*C, mu, *df, true);
    gemv(-2.0, U, *c, 1.0, *df, 'T');

    /* function value */
    f = dot(*b, *b);

    diff = f0 - f;
//    std::cerr << n << ": f=" << f;
    if (diff > 0.0) {
//      std::cerr << " (accept)" << std::endl;
      f0 = f;
      *df0 = *df;
      *mu0 = mu;
    } else {
//      std::cerr << " (reject)" << std::endl;
      delta *= 0.5;
    }
    ++n;
  } while ((diff < 0.0 || diff/f0 >= 1.0e-6) && n < 100);
  mu = *mu0;

//  for (int i = 0; i < mu.size(); ++i) {
//    std::cerr << mu(i) << ' ';
//  }
//  std::cerr << std::endl;

  synchronize();
  delete lv;
  delete a;
  delete b;
  delete c;
  delete df;
  delete df0;
  delete mu0;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class M1, class V1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::minimise(M1& U, V1& lw1s) {
  /* pre-condition */
  assert (U.size1() == lw1s.size());

  const int P = U.size1();
  const int N = U.size2();

  static const bi::Location L = (bi::Location)M1::on_device;
  typedef real temp_value_type;
  typedef typename locatable_vector<L,real>::type temp_vector_type;
  typedef typename locatable_matrix<L,real>::type temp_matrix_type;

  temp_vector_type g(P), h(P), b(P), w(P), v(P);
  temp_matrix_type U2(P, N);
  temp_matrix_type dF(P, 2*N);
  temp_matrix_type dV(P, 2*N);
  temp_matrix_type dH(P, 2*N);
  temp_matrix_type A(P, 2*N);
  temp_vector_type mu(N), sigma(N), inv_sigma(N), inv_sigma0(N), df(2*N), df0(2*N), mu0(N), sigma0(N);

  /* ranges */
  BOOST_AUTO(mu_stuttered, make_stuttered_range(mu.begin(), mu.end(), P));
  BOOST_AUTO(inv_sigma_stuttered, make_stuttered_range(inv_sigma.begin(), inv_sigma.end(), P));
  BOOST_AUTO(h_repeated, make_repeated_range(h.begin(), h.end(), N));
  BOOST_AUTO(b_repeated, make_repeated_range(b.begin(), b.end(), N));
  BOOST_AUTO(w_repeated, make_repeated_range(w.begin(), w.end(), N));

  real det, f, f0, delta, diff;
  real lhat = bi::sum_exp(lw1s.begin(), lw1s.end(), REAL(0.0))/P;
  int n;

  mu.clear();
  bi::fill(inv_sigma.begin(), inv_sigma.end(), 1.0);
  element_exp(lw1s.begin(), lw1s.end(), w.begin());
  dot_rows(U, g);
  n = 0;
  do {
    /* take step */
    if (n > 0) {
      mu = mu0;
      inv_sigma = inv_sigma0;
      axpy(-delta, subrange(df0, 0, N), mu);
      axpy(-delta, subrange(df0, N, N), inv_sigma);
    }

    /* det */
    det = 1.0/bi::prod(inv_sigma.begin(), inv_sigma.end(), 1.0);

    /* A */
    MAKE_ARGS(argsA, U.begin(), mu_stuttered.begin(), inv_sigma_stuttered.begin());
    thrust::transform(argsA, argsA + N*P, A.begin(), disturbance_compute_A<temp_value_type>());

    /* h */
    dot_rows(A, h);
    axpy(-1.0, g, h);
    scal(0.5, h);

    /* v */
    MAKE_ARGS(argsv, h.begin());
    thrust::transform(argsv, argsv + P, v.begin(), disturbance_compute_v<temp_value_type>(det));

    /* dH */
    MAKE_ARGS(argsdH1, A.begin(), inv_sigma_stuttered.begin());
    thrust::transform(argsdH1, argsdH1 + N*P, dH.begin(), disturbance_compute_dH1<temp_value_type>());

    MAKE_ARGS(argsdH2, A.begin(), U.begin(), mu_stuttered.begin());
    thrust::transform(argsdH2, argsdH2 + N*P, columns(dH, N, N).begin(), disturbance_compute_dH2<temp_value_type>());

    /* dV */
    MAKE_ARGS(argsdV1, dH.begin(), h_repeated.begin());
    thrust::transform(argsdV1, argsdV1 + N*P, dV.begin(), disturbance_compute_dV1<temp_value_type>(det));

    MAKE_ARGS(argsdV2, dH.begin(), h_repeated.begin(), inv_sigma_stuttered.begin());
    thrust::transform(argsdV2, argsdV2 + N*P, columns(dV, N, N).begin(), disturbance_compute_dV2<temp_value_type>(det));

    /* b */
    MAKE_ARGS(argsb, v.begin(), w.begin());
    thrust::transform(argsb, argsb + P, b.begin(), disturbance_compute_b<temp_value_type>(lhat));

    /* dF */
    MAKE_ARGS(argsdF, b_repeated.begin(), w_repeated.begin(), dV.begin())
    thrust::transform(argsdF, argsdF + 2*N*P, dF.begin(), disturbance_compute_dF<temp_value_type>());

    /* df */
    sum_rows(dF, df);

    /* f */
    f = dot(b, b);

//    std::cerr << n << ": f=" << f;
    if (n == 0) {
//      std::cerr << " (accept)" << std::endl;
      f0 = f;
      df0 = df;
      mu0 = mu;
      inv_sigma0 = inv_sigma;
      delta = 1.0/f0;
    } else {
      diff = f0 - f;
      if (diff > 0.0) {
//        std::cerr << " (accept)" << std::endl;
        f0 = f;
        df0 = df;
        mu0 = mu;
        inv_sigma0 = inv_sigma;
      } else {
//        std::cerr << " (reject)" << std::endl;
        delta *= 0.5;
      }
    }
    ++n;
  } while ((n == 1 || diff < 0.0 || diff/f0 >= 1.0e-6) && n < 100);

  thrust::transform(inv_sigma.begin(), inv_sigma.end(), sigma.begin(), pow_constant_functor<temp_value_type>(-2.0));

  /* sample new random variates */
  this->rng.gaussians(matrix_as_vector(U2));
  set_rows(U, mu);
  gdmm(1.0, sigma, U2, 1.0, U, 'R');

  /* weight adjustment */
  dot_rows(U2, h);
  axpy(-1.0, g, h);
  axpy(0.5, h, lw1s);
  thrust::transform(lw1s.begin(), lw1s.end(), lw1s.begin(), add_constant_functor<real>(log(det)));
}

#endif
