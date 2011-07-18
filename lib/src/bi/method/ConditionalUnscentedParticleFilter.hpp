/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1757 $
 * $Date: 2011-07-14 13:08:12 +0800 (Thu, 14 Jul 2011) $
 */
#ifndef BI_METHOD_CONDITIONALUNSCENTEDPARTICLEFILTER_HPP
#define BI_METHOD_CONDITIONALUNSCENTEDPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "UnscentedKalmanFilter.hpp"

namespace bi {
/**
 * Particle filter with unscented Kalman filter proposals conditioned on each
 * particle.
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
class ConditionalUnscentedParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL,SH>,
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
  ConditionalUnscentedParticleFilter(B& m, Random& rng, const real delta = 1.0,
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
   * Construct proposals for upcoming prediction using unscented Kalman
   * filters.
   *
   * @tparam L1 Location.
   * @tparam L2 Location.
   *
   * @param T Time to which to predict.
   * @param[in,out] theta1 Static state of PF.
   * @param[in,out] s1 State of PF.
   * @param[out] theta2 Static state for UKF.
   * @param[out] s2 State for UKF.
   */
  template<Location L1, Location L2>
  void propose(const real T, Static<L1>& theta1, State<L1>& s1,
      Static<L2>& theta2, State<L2>& s2);

  /**
   * Construct proposals for upcoming prediction using unscented Kalman
   * filters, from fixed starting state.
   *
   * @tparam V1 Vector type.
   * @tparam L1 Location.
   * @tparam L2 Location.
   *
   * @param T Time to which to predict.
   * @param x0 Starting state.
   * @param[in,out] theta1 Static state of PF.
   * @param[in,out] s1 State of PF.
   * @param[out] theta2 Static state for UKF.
   * @param[out] s2 State for UKF.
   */
  template<class V1, Location L1, Location L2>
  void propose(const real T, const V1 x0, Static<L1>& theta1, State<L1>& s1,
      Static<L2>& theta2, State<L2>& s2);

  /**
   * Sample noise terms for upcoming prediction.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] lws Log-weights.
   */
  template<class V1>
  void sample(V1 lws);
  //@}

  using particle_filter_type::getOutput;
  using particle_filter_type::getTime;
  using particle_filter_type::summarise;
  using particle_filter_type::sampleTrajectory;
  using particle_filter_type::correct;
  using particle_filter_type::resample;
  using particle_filter_type::output;
  using particle_filter_type::flush;
  using particle_filter_type::init;
  using particle_filter_type::term;

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
   * Number of particles in particle filter.
   */
  int P1;

  /**
   * Number of sigma points in unscented Kalman filters (per particle).
   */
  int P2;

  /**
   * Number of observations.
   */
  int W;

  /**
   * Number of noise terms.
   */
  int V;

  int nupdates;

  /**
   * \f$\hat{\boldsymbol{\mu}}^1,\ldots,\hat{\boldsymbol{\mu}^P}\f$;
   * proposal means from UKFs.
   */
  matrix_type muU;

  /**
   * \f$\hat{U}^1,\ldots,\hat{U}^P\f$; proposal Cholesky factors from UKFs.
   */
  matrix_type RU;

  /**
   * \f$|\hat{U}^1|,\ldots,\hat{U}^P\f$; proposal log-determinants from UKF.
   */
  vector_type ldetRU;

  /**
   * Estimate parameters as well as state?
   */
  static const bool haveParameters = SH == STATIC_OWN;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
  static const int M = ND + NC + NR + ((SH == STATIC_OWN) ? NP : 0);
};

/**
 * Factory for creating ConditionalUnscentedParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ConditionalUnscentedParticleFilter
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct ConditionalUnscentedParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return ConditionalUnscentedParticleFilter object. Caller has ownership.
   *
   * @see ConditionalUnscentedParticleFilter::ConditionalUnscentedParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(B& m,
      Random& rng, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in,
        obs, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::ConditionalUnscentedParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO3* out) :
    particle_filter_type(m, rng, delta, in, obs, out),
    kalman_filter_type(m, delta, (in == NULL) ? NULL : new IO1(*in),
        (obs == NULL) ? NULL : new IO2(*obs)) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam, const real relEss) {
  /* pre-conditions */
  assert (T > particle_filter_type::state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);

  typedef typename locatable_vector<ON_HOST,real>::type V2;
  typedef typename locatable_matrix<ON_HOST,real>::type M2;
  typedef typename locatable_vector<L,real>::type V3;
  typedef typename locatable_vector<L,int>::type V4;

  /* ukf temps */
  Static<L> theta1(particle_filter_type::m, theta.size());
  State<L> s1(particle_filter_type::m);

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* filter */
  init(theta, lws, as);
  int n = 0, r = 0;
  while (particle_filter_type::getTime() < T) {
    propose(T, theta, s, theta1, s1);
    sample(lws);
    particle_filter_type::predict(T, theta, s);
    particle_filter_type::correct(s, lws);
    particle_filter_type::output(n, theta, s, r, lws, as);
    ++n;
    r = particle_filter_type::state.t < T && resample(theta, s, lws, as, resam, relEss);
  }

  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R, class V1>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    const V1 x0, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T > particle_filter_type::state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);
  assert (x0.size() == ND + NC + NP);





}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam,
    const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  particle_filter_type::reset();
  kalman_filter_type::reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L1, bi::Location L2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::propose(
    const real T, Static<L1>& theta1, State<L1>& s1, Static<L2>& theta2,
    State<L2>& s2) {
  /* pre-condition */
  assert (!haveParameters);

  typedef typename locatable_temp_vector<L2,real>::type V1;
  typedef typename locatable_temp_matrix<L2,real>::type M1;

  real tj = kalman_filter_type::obs(T, s2);
  kalman_filter_type::parameters(tj);

  nupdates = kalman_filter_type::nupdates;
  P1 = s1.size(); // no. particles in pf
  P2 = kalman_filter_type::P; // no. sigma points in ukf per particle
  W = kalman_filter_type::W;
  V = NR*nupdates;

  if (nupdates > 0) {
    const int N1 = kalman_filter_type::N1;
    const int N2 = kalman_filter_type::N2;
    const int nextra = kalman_filter_type::nextra;
    const real Wc0 = kalman_filter_type::Wc0;
    const real Wi = kalman_filter_type::Wi;
    BOOST_AUTO(&Wm, kalman_filter_type::Wm);

    M1 muY(W, P1); // means of observations
    M1 SigmaY(W, P1*W); // observation covariances
    M1 SigmaUY(V, P1*W); // noise-observation cross-covariances
    M1 RY(W, P1*W); // observation covariance Cholesky factors
    M1 X1(P1*P2, N2), X2(P1*P2, N2); // matrices of sigma points
    M1 tmp1(W, P1), tmp2(V, P1); // workspaces
    M1 Z(P1, ND + NC + NR);

    muU.resize(V, P1, false); // means of noise
    RU.resize(V, P1*V, false); // noise covariance Cholesky factors
    ldetRU.resize(P1, false); // log-determinants of noise covariance Cholesky factors

    int p, w;

    /* construct and propagate sigma points */
    columns(Z, 0, ND) = s1.get(D_NODE);
    columns(Z, ND, NC) = s1.get(C_NODE);
    columns(Z, ND + NC, NR) = s1.get(R_NODE);
    log_columns(columns(Z, 0, ND), kalman_filter_type::m.getLogs(D_NODE));
    log_columns(columns(Z, ND, NC), kalman_filter_type::m.getLogs(C_NODE));
    for (p = 0; p < P1; ++p) {
      set_rows(subrange(X1, p*P2, P2, 0, ND + NC + NR), row(Z, p));
    }
    s2.resize(P1*P2, false);
    theta2.resize(1);
    theta2 = theta1;
    kalman_filter_type::advanceNoise(tj, X1.ref(), X2.ref(), theta2, s2);

    /* observation */
    BOOST_AUTO(y, duplicate_vector(row(s2.get(OY_NODE), 0)));
    log_vector(*y, kalman_filter_type::oLogs);
    set_columns(tmp1, *y);
    delete y;

    #pragma omp parallel
    {
      int offset, p;

      /* observation means */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(Y, subrange(X2, p*P2, P2, N2 - W, W));
        BOOST_AUTO(mu, column(muY, p));

        gemv(1.0, Y, Wm, 0.0, mu, 'T');
      }

      /* observation covariances */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(Y, subrange(X2, p*P2, P2, N2 - W, W));
        BOOST_AUTO(mu, column(muY, p));
        BOOST_AUTO(Sigma, columns(SigmaY, p*W, W));

        sub_rows(Y, mu);
        syrk(Wi, rows(Y, 1, 2*N1), 0.0, Sigma, 'U', 'T');
        syr(Wc0, row(Y, 0), Sigma, 'U');
      }

      /* noise-observation cross-covariances */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        offset = ND + NC + NR + NR*(nextra - nupdates);
        BOOST_AUTO(Y1, subrange(X2, p*P2 + 1 + offset, V, N2 - W, W));
        BOOST_AUTO(Y2, subrange(X2, p*P2 + 1 + N1 + offset, V, N2 - W, W));
        BOOST_AUTO(U1, subrange(X2, p*P2 + 1 + offset, V, offset, V));
        BOOST_AUTO(U2, subrange(X2, p*P2 + 1 + N1 + offset, V, offset, V));
        BOOST_AUTO(Sigma, columns(SigmaUY, p*W, W));

        gdmm(Wi, diagonal(U1), Y1, 0.0, Sigma);
        gdmm(Wi, diagonal(U2), Y2, 1.0, Sigma);
      }

      /* compute conditional means and covariance Cholesky factors */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(SigmaY1, columns(SigmaY, p*W, W));
        BOOST_AUTO(SigmaUY1, columns(SigmaUY, p*W, W));
        BOOST_AUTO(RU1, columns(RU, p*V, V));
        BOOST_AUTO(RY1, columns(RY, p*W, W));
        BOOST_AUTO(muU1, column(muU, p));

        try {
          /* conditional mean */
          axpy(-1.0, column(muY, p), column(tmp1, p));
          chol(SigmaY1, RY1, 'U');
          potrs(RY1, columns(tmp1, p, 1), 'U');
          gemv(1.0, SigmaUY1, column(tmp1, p), 0.0, muU1);

          /* conditional covariance Cholesky factor */
          trsm(1.0, RY1, SigmaUY1, 'R', 'U');
          ident(RU1);
          for (w = 0; w < W; ++w) {
            ch1dn(RU1, column(SigmaUY1, w), column(tmp2, p));
          }
          ldetRU(p) = log(bi::prod(diagonal(RU1).begin(), diagonal(RU1).end(), 1.0));
        } catch (Exception e) {
          ident(RU1);
          muU1.clear();
          ldetRU(p) = 0.0;
        }
      }

//      if (doLookahead) {
//        /* compute stage-1 weights */
//        dot_columns(tmp1, lw1);
//        for (p = 0; p < P1; ++p) {
//          ldetY(p) = log(bi::prod(diagonal(RY1).begin(), diagonal(RY1).end(), 1.0));
//        }
//        scal(-0.5, lw1);
//        axpy(-1.0, ldetY, lw1);
//      }
    }
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class V1>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::sample(
    V1 lws) {
  if (nupdates > 0) {
    V1 lw1(P1), lw2(P1); // weight correction vectors
    BOOST_AUTO(&U, particle_filter_type::rUpdater.buf());

    U.resize(P1, V);
    particle_filter_type::rng.gaussians(vec(U));
    dot_rows(U, lw1);

    #pragma omp parallel
    {
      int p;
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(RU1, columns(RU, p*V, V));
        BOOST_AUTO(muU1, column(muU, p));
        BOOST_AUTO(u1, row(U, p));

        trmv(RU1, u1, 'U');
        axpy(1.0, muU1, u1);
      }
      particle_filter_type::rUpdater.setNext(V/NR);

      /* weight correct */
      dot_rows(U, lw2);
      axpy(0.5, lw1, lws);
      axpy(-0.5, lw2, lws);
      axpy(1.0, ldetRU, lws);
    }
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class V1, bi::Location L1, bi::Location L2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::propose(
    const real T, const V1 x0, Static<L1>& theta1, State<L1>& s1,
    Static<L2>& theta2, State<L2>& s) {

}

#endif
