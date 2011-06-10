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
class DisturbanceParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL,SH> {
public:
  /**
   * Constructor.
   *
   * @tparam IO4 #concept::UnscentedKalmanFilterBuffer type.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param delta Time step for d- and r-nodes.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  template<class IO4>
  DisturbanceParticleFilter(B& m, Random& rng, const real delta = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO4* proposal = NULL, IO3* out = NULL);

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
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL, const real relEss = 1.0);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * @copydoc ParticleFilter::init()
   */
  template<Location L, class V1, class V2>
  void init(Static<L>& theta, V1& lws, V2& as);

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
  //@}

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
   * Build parameter-independent parts of proposal.
   *
   * @tparam IO4 #concept::UnscentedKalmanFilterBuffer type.
   *
   * @param proposal Output of UnscentedKalmanFilter to use as proposal.
   */
  template<class IO4>
  void prepare(IO4* proposal);

  /**
   * \f$\boldsymbol{\mu}_t\f$; proposal mean vector at each time.
   */
  matrix_type mut;

  /**
   * \f$U_t\f$; proposal standard deviation matrix at each time.
   */
  matrix_type Ut;

  /**
   * \f$K_t\f$; gain matrix at each time.
   */
  matrix_type Kt;

  /**
   * \f$\mathbf{a}_t = \boldsymbol{\mu}_u - K_t\boldsymbol{\mu}_{\theta}\f$;
   * mean offset vector at each time.
   */
  matrix_type at;

  /**
   * \f$|U_t|\f$; determinants of proposal covariance at each time.
   */
  host_vector<real> detUt;

  /**
   * Current index into proposals.
   */
  int k;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
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
  template<class B, class IO1, class IO2, class IO3, class IO4>
  static DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(B& m,
      Random& rng, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO4* proposal = NULL, IO3* out = NULL) {
    return new DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in,
        obs, proposal, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class IO4>
bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::DisturbanceParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO4* proposal,
    IO3* out) :
    ParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in, obs, out) {
  prepare(proposal);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam, const real relEss) {
  /* pre-conditions */
  assert (T > this->state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);

  int n = 0, r = 0;

  typename locatable_temp_vector<L,real>::type lws(s.size());
  typename locatable_temp_vector<L,int>::type as(s.size());

  init(theta, lws, as);
  while (this->state.t < T) {
    predict(T, theta, s, lws);
    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = this->state.t < T && resample(theta, s, lws, as, resam, relEss);
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T > this->state.t);
  assert (this->out != NULL);
  assert (relEss >= 0.0 && relEss <= 1.0);

  synchronize();

  int n = 0, r = 0, a = 0;

  typename locatable_temp_vector<L,real>::type lws(s.size());
  typename locatable_temp_vector<L,int>::type as(s.size());

  init(theta, lws, as);
  while (this->state.t < T) {
    predict(T, theta, s, lws);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_NODE), 0) = column(xd, n);
    row(s.get(C_NODE), 0) = column(xc, n);
    row(s.get(R_NODE), 0) = column(xr, n);

    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = this->state.t < T && resample(theta, s, a, lws, as, resam, relEss);
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::predict(
    const real tnxt, Static<L>& theta, State<L>& s, V1& lws) {
  BOOST_AUTO(lw1, temp_vector<V1>(lws.size()));
  BOOST_AUTO(lw2, temp_vector<V1>(lws.size()));
  BOOST_AUTO(X, s.get(R_NODE));

  real to = tnxt;
  if (this->oyUpdater.hasNext() && this->oyUpdater.getNextTime() >= this->getTime() && this->oyUpdater.getNextTime() < to) {
    to = this->oyUpdater.getNextTime();
  }

  /* simulate forward */
  if (this->haveParameters) {
    this->sim.init(theta); // p- and s-nodes need updating
  }
  while (this->state.t < to) {
    /* propose */
    this->rng.gaussians(matrix_as_vector(X));
    dot_rows(X, *lw1);
    trmm(1.0, columns(Ut, k*NR, NR), X, 'R', 'U');
    add_rows(X, column(mut, k));
    dot_rows(X, *lw2);
    this->rUpdater.skipNext();
    ++k;

    /* correct weights */
    thrust::transform(lws.begin(), lws.end(), lws.begin(),
        add_constant_functor<real>(log(detUt(k))));
    axpy(0.5, *lw1, lws);
    axpy(-0.5, *lw2, lws);

    /* propagate */
    this->sim.advance(to, s);
    this->state.t = this->sim.getTime();
  }

  synchronize();
  delete lw1;
  delete lw2;

  /* post-condition */
  assert (this->sim.getTime() == this->state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class IO4>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::prepare(IO4* proposal) {
  const int T = proposal->size2();
  int t;

  mut.resize(NR, T);
  Ut.resize(NR, NR*T);
  Kt.resize(NR*T, NP);
  at.resize(NR, T);
  detUt.resize(T);

  ExpGaussianPdf<> pu(NR);
  ExpGaussianPdf<> ptheta(NP);
  ExpGaussianPdf<> q(NR);

  host_vector<> mu(ND + NC + NR + NP);
  host_matrix<> Sigma(ND + NC + NR + NP, ND + NC + NR + NP);
  host_matrix<> C(NR, NP);

  ptheta.setLogs(this->m.getLogs(P_NODE));

  for (t = 0; t < T; ++t) {
    proposal->readCorrectedState(t, mu, Sigma);

    pu.mean() = subrange(mu, ND + NC, NR);
    pu.cov() = subrange(Sigma, ND + NC, NR, ND + NC, NR);
    pu.init();

    ptheta.mean() = subrange(mu, ND + NC + NR, NP);
    ptheta.cov() = subrange(Sigma, ND + NC + NR, NP, ND + NC + NR, NP);
    ptheta.init();

    C = subrange(Sigma, ND + NC, NR, ND + NC + NR, NP);
    condition(pu, ptheta, C, ptheta.mean(), q);

    /* precomputes for this time */
    columns(Ut, t*NR, NR) = q.std();
    symm(1.0, ptheta.prec(), C, 0.0, rows(Kt, t*NR, NR), 'R', 'U');
    column(at, t) = pu.mean();
    gemv(-1.0, rows(Kt, t*NR, NR), ptheta.mean(), 1.0, column(at, t));
    detUt(t) = q.det();
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(
    Static<L>& theta, V1& lws, V2& as) {
  ParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(theta, lws, as);

  /* parameter-dependent parts of proposal */
  BOOST_AUTO(x, host_temp_vector<real>(NP));
  *x = row(theta.get(P_NODE), 0);
  log_vector(*x, this->m.getLogs(P_NODE));

  k = 0;
  mut = at;
  gemv(1.0, Kt, *x, 1.0, matrix_as_vector(mut));
  synchronize();

  delete x;
}

#endif
