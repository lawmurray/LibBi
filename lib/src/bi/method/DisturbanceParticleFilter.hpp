/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1246 $
 * $Date: 2011-01-31 16:23:46 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP
#define BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "UnscentedKalmanFilter.hpp"

namespace bi {
/**
 * Disturbance particle filter with unscented Kalman filter proposal.
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
   * distribution for particle filter.
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
   * and use to construct proposal distribution for particle filter.
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
   * Predict.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time where observations are available is reached.
   */
  template<Location L, class V1>
  void predict(const real tnxt, Static<L>& theta, State<L>& s, V1& lws);

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
   * Vector type for proposal.
   */
  typedef typename locatable_vector<CL,real>::type vector_type;

  /**
   * Matrix type for proposal.
   */
  typedef typename locatable_matrix<CL,real>::type matrix_type;

  /**
   * \f$\boldsymbol{\mu}_t\f$; proposal mean at current time.
   */
  vector_type mu;

  /**
   * \f$U_t\f$; proposal standard deviation at current time.
   */
  matrix_type U;

  /**
   * \f$|U_t|\f$; determinant of proposal standard deviation at current time.
   */
  real detU;

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
    kalman_filter_type(m, delta, in, obs), mu(NR), U(NR,NR) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam, const real relEss) {
  /* pre-conditions */
  assert (T > particle_filter_type::state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);

  typedef typename locatable_vector<ON_HOST,real>::type V1;
  typedef typename locatable_matrix<ON_HOST,real>::type M1;
  typedef typename locatable_vector<L,real>::type V2;
  typedef typename locatable_vector<L,int>::type V3;

  /* ukf temps */
  Static<ON_HOST> theta1(particle_filter_type::m, theta.size());
  State<ON_HOST> s1(particle_filter_type::m);
  ExpGaussianPdf<V1,M1> corrected(M);
  ExpGaussianPdf<V1,M1> uncorrected(M);
  ExpGaussianPdf<V1,M1> observed(0);
  M1 SigmaXX(M, M), SigmaXY(M, 0);

  /* pf temps */
  V2 lws(s.size());
  V3 as(s.size());
  int n = 0, r = 0;
  theta1 = theta;

  init(theta, lws, as, theta1, corrected);
  while (particle_filter_type::state.t < T) {
    lookahead(T, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    predict(T, theta, s, lws);
    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = particle_filter_type::state.t < T && resample(theta, s, lws, as, resam, relEss);
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
  int n = 0, r = 0;
  theta1 = theta;

  /* initialise pf from fixed starting state */
  set_rows(s.get(D_NODE), subrange(x0, 0, ND));
  set_rows(s.get(C_NODE), subrange(x0, ND, NC));
  set_rows(theta.get(P_NODE), subrange(x0, ND + NC, NP));

  /* filter */
  init(theta, lws, as, theta1, corrected);
  while (particle_filter_type::getTime() < T) {
    if (n == 0) {
      lookahead(T, x0, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    } else {
      lookahead(T, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
    }
    predict(T, theta, s, lws);
    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = particle_filter_type::state.t < T && resample(theta, s, lws, as, resam, relEss);
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
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L1, class V1, class V2, bi::Location L2, class V3,
    class M3>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(
    Static<L1>& theta, V1& lws, V2& as, Static<L2>& theta1,
    ExpGaussianPdf<V3,M3>& corrected) {
  particle_filter_type::init(theta, lws, as);
  kalman_filter_type::init(theta1, corrected);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::lookahead(
    const real tnxt, ExpGaussianPdf<V1,M1>& corrected, Static<L>& theta1,
    State<L>& s1, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  particle_filter_type::mark();
  kalman_filter_type::predict(tnxt, corrected, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);
  kalman_filter_type::correct(uncorrected, SigmaXY, s1, observed, corrected);
  particle_filter_type::restore();

  mu = subrange(corrected.mean(), ND + NC, NR);
  potrf(subrange(corrected.cov(), ND + NC, NR, ND + NC, NR), U);
  detU = bi::prod(diagonal(U).begin(), diagonal(U).end(), 1.0);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class V2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::lookahead(
    const real tnxt, const V2 x0, ExpGaussianPdf<V1,M1>& corrected,
    Static<L>& theta1, State<L>& s1, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  particle_filter_type::mark();
  kalman_filter_type::predict(tnxt, x0, theta1, s1, observed, uncorrected, SigmaXX, SigmaXY);

  if (kalman_filter_type::getTime() > particle_filter_type::getTime()) {
    kalman_filter_type::correct(uncorrected, SigmaXY, s1, observed, corrected);
    mu = subrange(corrected.mean(), ND + NC, NR);
    potrf(subrange(corrected.cov(), ND + NC, NR, ND + NC, NR), U);
    detU = bi::prod(diagonal(U).begin(), diagonal(U).end(), 1.0);
  } else {
    mu.clear();
    ident(U);
    detU = 1.0;
  }
  particle_filter_type::restore();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::predict(
    const real tnxt, Static<L>& theta, State<L>& s, V1& lws) {
  /* pre-condition */
  assert (s.size() == lws.size());

  /* propose */
  BOOST_AUTO(lw1, temp_vector<V1>(lws.size()));
  BOOST_AUTO(lw2, temp_vector<V1>(lws.size()));
  BOOST_AUTO(X, s.get(R_NODE));

  particle_filter_type::rng.gaussians(matrix_as_vector(X));
  dot_rows(X, *lw1);
  trmm(1.0, U, X, 'R', 'U');
  add_rows(X, mu);
  dot_rows(X, *lw2);
  particle_filter_type::rUpdater.skipNext();

  /* correct weights */
  thrust::transform(lws.begin(), lws.end(), lws.begin(), add_constant_functor<real>(log(detU)));
  axpy(0.5, *lw1, lws);
  axpy(-0.5, *lw2, lws);

  /* propagate */
  particle_filter_type::predict(tnxt, theta, s);

  synchronize();
  delete lw1;
  delete lw2;

  /* post-condition */
  assert (particle_filter_type::getTime() == kalman_filter_type::getTime());
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L1, bi::Location L2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::term(
    Static<L1>& theta, Static<L2>& theta1) {
  particle_filter_type::term(theta);
  kalman_filter_type::term(theta1);
}

#endif
