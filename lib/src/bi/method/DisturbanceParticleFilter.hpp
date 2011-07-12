/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP
#define BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "UnscentedKalmanFilter.hpp"

namespace bi {
/**
 * Disturbance particle filter with marginal unscented Kalman filter
 * proposal.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::ParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST,
    StaticHandling SH = STATIC_SHARED>
class DisturbanceParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL,SH>,
    private UnscentedKalmanFilter<B,IO1,IO2,IO3,ON_HOST,SH> {
public:
  /**
   * Particle filter type.
   */
  typedef ParticleFilter<B,IO1,IO2,IO3,CL,SH> particle_filter_type;

  /**
   * Kalman filter type.
   */
  typedef UnscentedKalmanFilter<B,IO1,IO2,IO3,ON_HOST,SH> kalman_filter_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param delta Time step for d- and r-nodes.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  DisturbanceParticleFilter(B& m, Random& rng, const real delta = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~DisturbanceParticleFilter();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc ParticleFilter::filter()
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL,
      const real relEss = 1.0);

  /**
   * @copydoc ParticleFilter::filter()
   */
  template<Location L, class R, class V1>
  void filter(const real T, const V1 x0, Static<L>& theta, State<L>& s,
      R* resam = NULL, const real relEss = 1.0);

  /**
   * @copydoc ParticleFilter::filter()
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL, const real relEss = 1.0);

  /**
   * @copydoc #concept::ParticleFilter::reset()
   */
  void reset();
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam L1 Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam L2 Location.
   * @tparam V3 Vector type.
   * @tparam M3 Matrix type.
   *
   * @param theta Static state for particle filter.
   * @param theta1 Static state for Kalman filter.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[out] corrected Prior over initial state for unscented Kalman
   * filter.
   */
  template<Location L1, class V1, class V2, Location L2, class V3, class M3>
  void init(Static<L1>& theta, V1& lws, V2& as, Static<L2>& theta1,
      ExpGaussianPdf<V3,M3>& corrected);

  /**
   * Lookahead using unscented Kalman filter and use to construct proposal
   * distributions for particle filter.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Time to which to predict.
   * @param corrected Corrected state marginal at current time.
   * @param[out] theta1 Static state for unscented Kalman filter.
   * @param[out] s1 State for unscented Kalman filter.
   * @param[out] observed Observation marginal at next time.
   * @param[out] uncorrected Uncorrected state marginal at next time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   *
   * Note that @p theta1 and @p s1 should be different to the analagous
   * arguments provided to other calls, or the particle filter's state
   * will be lost!
   */
  template<Location L, class V1, class M1>
  void lookahead(const real tnxt, ExpGaussianPdf<V1,M1>& corrected,
      Static<L>& theta1, State<L>& s1, ExpGaussianPdf<V1,M1>& observed,
      ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY);

  /**
   * Lookahead using unscented Kalman filter from fixed starting state
   * and use to construct proposal distributions for particle filter.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam V2 Vector type.
   *
   * @param tnxt Time to which to predict.
   * @param x0 Starting state.
   * @param corrected[out] Corrected state marginal at current time.
   * @param[out] theta1 Static state for unscented Kalman filter.
   * @param[out] s1 State for unscented Kalman filter.
   * @param[out] observed Observation marginal at next time.
   * @param[out] uncorrected Uncorrected state marginal at next time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   *
   * Note that @p theta1 and @p s1 should be different to the analagous
   * arguments provided to other calls, or the particle filter's state
   * will be lost!
   */
  template<Location L, class V1, class M1, class V2>
  void lookahead(const real tnxt, const V2 x0,
      ExpGaussianPdf<V1,M1>& corrected, Static<L>& theta1, State<L>& s1,
      ExpGaussianPdf<V1,M1>& observed, ExpGaussianPdf<V1,M1>& uncorrected,
      M1& SigmaXX, M1& SigmaXY);

  /**
   * Propose stochastic terms for next particle filter prediction.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] lws Log-weights.
   */
  template<class V1>
  void propose(V1 lws);

  /**
   * Clean up.
   *
   * @tparam L1 Location.
   * @tparam L2 Location.
   *
   * @param theta Static state for particle filter.
   * @param theta1 Static state for Kalman filter.
   */
  template<Location L1, Location L2>
  void term(Static<L1>& theta, Static<L2>& theta1);
  //@}

  using particle_filter_type::getOutput;
  using particle_filter_type::getTime;
  using particle_filter_type::summarise;
  using particle_filter_type::sampleTrajectory;
  using particle_filter_type::correct;
  using particle_filter_type::resample;
  using particle_filter_type::output;
  using particle_filter_type::flush;

private:
  /**
   * Vector type for proposals.
   */
  typedef typename locatable_temp_vector<ON_HOST,real>::type vector_type;

  /**
   * Matrix type for proposals.
   */
  typedef typename locatable_temp_matrix<ON_HOST,real>::type matrix_type;

  /**
   * \f$\hat{\boldsymbol{\mu}}_1,\ldots,\hat{\boldsymbol{\mu}}\f$; noise term
   * proposal mean from UKF.
   */
  std::vector<vector_type*> mu;

  /**
   * \f$\hat{U}_1,\ldots,\hat{U}_T\f$; noise term proposal Cholesky factors
   * from UKF.
   */
  std::vector<matrix_type*> U;

  /**
   * \f$|\hat{U}_1|,\ldots,\hat{U}_T\f$; noise term proposal determinants
   * from UKF.
   */
  std::vector<real> detU;

  /**
   * Progress of PF.
   */
  int k1;

  /**
   * Progress of UKF.
   */
  int k2;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
  static const int M = ND + NC + NR + ((SH == STATIC_OWN) ? NP : 0);
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
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct DisturbanceParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return DisturbanceParticleFilter object. Caller has ownership.
   *
   * @see DisturbanceParticleFilter::DisturbanceParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(B& m,
      Random& rng, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in,
        obs, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::DisturbanceParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO3* out) :
    particle_filter_type(m, rng, delta, in, obs, out),
    kalman_filter_type(m, delta, (in == NULL) ? NULL : new IO1(*in),
        (obs == NULL) ? NULL : new IO2(*obs)),
    mu(out->size2(), NULL), U(out->size2(), NULL), detU(out->size2(), 1.0) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::~DisturbanceParticleFilter() {
  BOOST_AUTO(iter1, mu.begin());
  for (; iter1 != mu.end(); ++iter1) {
    delete *iter1;
  }
  BOOST_AUTO(iter2, U.begin());
  for (; iter2 != U.end(); ++iter2) {
    delete *iter2;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam, const real relEss) {
  /* pre-conditions */
  assert (T > particle_filter_type::state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);

  typedef typename locatable_vector<ON_HOST,real>::type V2;
  typedef typename locatable_matrix<ON_HOST,real>::type M2;
  typedef typename locatable_vector<L,real>::type V3;
  typedef typename locatable_vector<L,int>::type V4;

  /* ukf temps */
  Static<ON_HOST> theta1(particle_filter_type::m, theta.size());
  State<ON_HOST> s1(particle_filter_type::m);
  ExpGaussianPdf<V2,M2> corrected(M);
  ExpGaussianPdf<V2,M2> uncorrected(M);
  ExpGaussianPdf<V2,M2> observed(0);
  M2 SigmaXX(M, M), SigmaXY(M, 0);

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* filter */
  init(theta, lws, as, theta1, corrected);
  #ifndef USE_CPU
  #pragma omp parallel sections
  #endif
  {
    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      lookahead(T, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    }

    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      int n = 0, r = 0;
      while (particle_filter_type::getTime() < T) {
        propose(lws);
        particle_filter_type::predict(T, theta, s);
        particle_filter_type::correct(s, lws);
        particle_filter_type::output(n, theta, s, r, lws, as);
        ++n;
        r = particle_filter_type::state.t < T && resample(theta, s, lws, as, resam, relEss);
      }
    }
  }

  synchronize();
  term(theta, theta1);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R, class V1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    const V1 x0, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T > particle_filter_type::state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);
  assert (x0.size() == ND + NC + NP);

  typedef typename locatable_vector<ON_HOST,real>::type V2;
  typedef typename locatable_matrix<ON_HOST,real>::type M2;
  typedef typename locatable_vector<L,real>::type V3;
  typedef typename locatable_vector<L,int>::type V4;

  /* ukf temps */
  Static<ON_HOST> theta1(particle_filter_type::m, theta.size());
  State<ON_HOST> s1(particle_filter_type::m);
  ExpGaussianPdf<V2,M2> corrected(M);
  ExpGaussianPdf<V2,M2> uncorrected(M);
  ExpGaussianPdf<V2,M2> observed(0);
  M2 SigmaXX(M, M), SigmaXY(M, 0);

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* initialise pf from fixed starting state */
  set_rows(s.get(D_NODE), subrange(x0, 0, ND));
  set_rows(s.get(C_NODE), subrange(x0, ND, NC));
  set_rows(theta.get(P_NODE), subrange(x0, ND + NC, NP));

  /* filter */
  init(theta, lws, as, theta1, corrected);
  #ifndef USE_CPU
  #pragma omp parallel sections
  #endif
  {
    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      lookahead(T, x0, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    }

    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      int n = 0, r = 0;
      while (particle_filter_type::getTime() < T) {
        propose(lws);
        particle_filter_type::predict(T, theta, s);
        particle_filter_type::correct(s, lws);
        particle_filter_type::output(n, theta, s, r, lws, as);
        ++n;
        r = particle_filter_type::state.t < T && resample(theta, s, lws, as, resam, relEss);
      }
    }
  }

  synchronize();
  term(theta, theta1);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam,
    const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  particle_filter_type::reset();
  kalman_filter_type::reset();
  k1 = 0;
  k2 = 0;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L1, class V1, class V2, bi::Location L2, class V3,
    class M3>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(
    Static<L1>& theta, V1& lws, V2& as, Static<L2>& theta1,
    ExpGaussianPdf<V3,M3>& corrected) {
  theta1 = theta;
  synchronize();

  particle_filter_type::init(theta, lws, as);
  kalman_filter_type::init(theta1, corrected);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::lookahead(
    const real T, ExpGaussianPdf<V1,M1>& corrected, Static<L>& theta1,
    State<L>& s1, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  real ti, tj;

  ti = kalman_filter_type::getTime();
  while (ti < T) {
    kalman_filter_type::predict(T, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    kalman_filter_type::correct(uncorrected, SigmaXY, s1, observed, corrected);
    tj = kalman_filter_type::getTime();

    real delta = kalman_filter_type::delta;
    int nextra = std::max(0, lt_steps(tj, delta) - le_steps(ti, delta));
    int nupdates = lt_steps(tj, delta) - lt_steps(ti, delta);

    /* allocate if necessary */
    if (this->mu[k2] == NULL) {
      this->mu[k2] = new vector_type(NR*nupdates);
    }
    if (this->U[k2] == NULL) {
      this->U[k2] = new matrix_type(NR*nupdates, NR*nupdates);
    }

    /* copy noise components from UKF time marginal */
    BOOST_AUTO(&mu, *this->mu[k2]);
    BOOST_AUTO(&U, *this->U[k2]);
    matrix_type Sigma(NR*nupdates, NR*nupdates);

    subrange(mu, 0, nextra*NR) = subrange(corrected.mean(), M, nextra*NR);
    subrange(Sigma, 0, nextra*NR, 0, nextra*NR) = subrange(corrected.cov(), M, nextra*NR, M, nextra*NR);
    if (nextra < nupdates) {
      subrange(mu, nextra*NR, NR) = subrange(corrected.mean(), ND + NC, NR);
      subrange(Sigma, nextra*NR, NR, nextra*NR, NR) = subrange(corrected.cov(), ND + NC, NR, ND + NC, NR);
      transpose(subrange(corrected.cov(), ND + NC, NR, M, nextra*NR), subrange(Sigma, 0, nextra*NR, nextra*NR, NR));
    }
    if (nupdates > 0) {
      potrf(Sigma, U, 'U');
    } else {
      ident(U);
    }
    detU[k2] = bi::prod(diagonal(U).begin(), diagonal(U).end(), 1.0);

    ti = tj;
    ++k2;
    #pragma omp flush(k2)
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class V2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::lookahead(
    const real T, const V2 x0, ExpGaussianPdf<V1,M1>& corrected,
    Static<L>& theta1, State<L>& s1, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  while (kalman_filter_type::getTime() < T) {
    if (kalman_filter_type::getTime() == 0.0) {
      kalman_filter_type::predict(T, x0, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    } else {
      kalman_filter_type::predict(T, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    }
    if (kalman_filter_type::getTime() > 0.0) {
      kalman_filter_type::correct(uncorrected, SigmaXY, s1, observed, corrected);
    }

    /* allocate if necessary */
    int N = corrected.size() - M + NR;
    if (this->mu[k2] == NULL) {
      this->mu[k2] = new vector_type(N);
    }
    if (this->U[k2] == NULL) {
      this->U[k2] = new matrix_type(N,N);
    }

    /* copy noise components from UKF time marginal */
    BOOST_AUTO(mu, *this->mu[k2]);
    BOOST_AUTO(U, *this->U[k2]);
    real& detU = this->detU[k2];

    subrange(mu, 0, N - NR) = subrange(corrected.mean(), M, N - NR);
    subrange(mu, N - NR, NR) = subrange(corrected.mean(), ND + NC, NR);
    subrange(U, 0, N - NR, 0, N - NR) = subrange(corrected.std(), M, N - NR, M, N - NR);
    subrange(U, N - NR, NR, N - NR, NR) = subrange(corrected.std(), ND + NC, NR, ND + NC, NR);
    detU = bi::prod(diagonal(U).begin(), diagonal(U).end(), 1.0);

    ++k2;
    #pragma omp flush(k2)
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class V1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::propose(V1 lws) {
  #ifndef USE_CPU
  /* ensure lookahead is actually ahead... */
  while (k1 >= k2) {
    #pragma omp flush(k2)
  }
  #endif

  const int P = lws.size();
  BOOST_AUTO(&mu, *this->mu[k1]);
  BOOST_AUTO(&U, *this->U[k1]);
  real detU = this->detU[k1];
  int nupdates = mu.size()/NR;
  BOOST_AUTO(&X, particle_filter_type::rUpdater.buf());
  BOOST_AUTO(lw1, temp_vector<V1>(lws.size()));
  BOOST_AUTO(lw2, temp_vector<V1>(lws.size()));

  X.resize(P, NR*nupdates, false);
  if (nupdates > 0) {
    /* propose */
    particle_filter_type::rng.gaussians(vec(X));
    dot_rows(X, *lw1);
    trmm(1.0, U, X, 'R', 'U');
    add_rows(X, mu);
    dot_rows(X, *lw2);

    /* correct weights */
    thrust::transform(lws.begin(), lws.end(), lws.begin(), add_constant_functor<real>(log(detU)));
    axpy(0.5, *lw1, lws);
    axpy(-0.5, *lw2, lws);
  }
  particle_filter_type::rUpdater.setNext(nupdates);
  ++k1;

  if (V1::on_device) {
    synchronize();
  }
  delete lw1;
  delete lw2;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L1, bi::Location L2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::term(
    Static<L1>& theta, Static<L2>& theta1) {
  particle_filter_type::term(theta);
  kalman_filter_type::term(theta1);
}

#endif
