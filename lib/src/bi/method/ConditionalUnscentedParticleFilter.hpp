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
    protected UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH> {
public:
  /**
   * Particle filter type.
   */
  typedef ParticleFilter<B,IO1,IO2,IO3,CL,SH> particle_filter_type;

  /**
   * Kalman filter type.
   */
  typedef UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH> kalman_filter_type;

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
   * @copydoc summarise_pf()
   */
  template<class T1, class V1, class V2>
  void summarise(T1* ll, V1* lls, V2* ess);

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
   * Perform precomputes for upcoming step.
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
  void prepare(const real T, Static<L1>& theta1, State<L1>& s1,
      Static<L2>& theta2, State<L2>& s2);

  /**
   * Perform precomputes for upcoming step, with fixed starting state.
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
  void prepare(const real T, const V1 x0, Static<L1>& theta1, State<L1>& s1,
      Static<L2>& theta2, State<L2>& s2);

  /**
   * Propose noise terms for upcoming prediction.
   *
   * @tparam V1 Integer vector type.
   * @tparam V2 Vector type.
   *
   * @param as Ancestors.
   * @param[in,out] lws Log-weights.
   */
  template<class V1, class V2>
  void propose(const V1 as, V2 lws);
  //@}

  using particle_filter_type::getOutput;
  using particle_filter_type::getTime;
  using particle_filter_type::summarise;
  using particle_filter_type::sampleTrajectory;
  using particle_filter_type::predict;
  using particle_filter_type::correct;
  using particle_filter_type::resample;
  using particle_filter_type::output;
  using particle_filter_type::flush;
  using particle_filter_type::init;
  using particle_filter_type::term;

protected:
  /**
   * Vector type for proposals.
   */
  typedef typename locatable_temp_vector<ON_HOST,real>::type vector_type;

  /**
   * Matrix type for proposals.
   */
  typedef typename locatable_temp_matrix<ON_HOST,real>::type matrix_type;

  using kalman_filter_type::N1;
  using kalman_filter_type::N2;
  using kalman_filter_type::nupdates;
  using kalman_filter_type::nextra;
  using kalman_filter_type::W;
  using kalman_filter_type::V;
  using kalman_filter_type::Wc0;
  using kalman_filter_type::Wi;
  using kalman_filter_type::Wm;
  using kalman_filter_type::m;
  using kalman_filter_type::oLogs;
  using kalman_filter_type::advanceNoise;
  using kalman_filter_type::obs;
  using kalman_filter_type::parameters;

  /**
   * Number of particles in particle filter.
   */
  int P1;

  /**
   * Number of sigma points in unscented Kalman filters (per particle).
   */
  int P2;

  /**
   * \f$\hat{\boldsymbol{\mu}_U}^1,\ldots,\hat{\boldsymbol{\mu}_U^P}\f$;
   * noise means from UKFs.
   */
  matrix_type muU;

  /**
   * \f$\hat{\boldsymbol{\mu}_Y}^1,\ldots,\hat{\boldsymbol{\mu}_Y^P}\f$;
   * observation means from UKFs.
   */
  matrix_type muY;

  /**
   * \f$\hat{\Sigma}_U^1,\ldots,\hat{\Sigma}_U^P\f$; noise covariances from
   * UKFs.
   *
   * @note Never explicitly needed, always \f$I\f$.
   */
  //matrix_type SigmaU;

  /**
   * \f$\hat{\Sigma}_Y^1,\ldots,\hat{\Sigma}_Y^P\f$; noise covariances from
   * UKFs.
   */
  matrix_type SigmaY;

  /**
   * \f$\hat{\Sigma}_{UY}^1,\ldots,\hat{\Sigma}_{UY}^P\f$; noise-observation
   * cross covariances from UKFs.
   */
  matrix_type SigmaUY;

  /**
   * \f$\hat{R}_U^1,\ldots,\hat{R}_U^P\f$; noise Cholesky factors from UKFs.
   */
  matrix_type RU;

  /**
   * \f$\hat{R}_Y^1,\ldots,\hat{R}_Y^P\f$; observation Cholesky factors from
   * UKFs.
   */
  matrix_type RY;

  /**
   * \f$|\hat{R}_U^1|,\ldots,|\hat{R}_U^P|\f$; log-determinants of noise
   * Cholesky factors.
   */
  vector_type ldetRU;

  /**
   * \f$|\hat{R}_Y^1|,\ldots,|\hat{R}_Y^P|\f$; log-determinants of observation
   * Cholesky factors.
   */
  vector_type ldetRY;

  /**
   * \f$\mathbf{y}_t - \hat{\boldsymbol{\mu}}_Y^1, \ldots, \mathbf{y}_t -
   * \hat{\boldsymbol{\mu}}_Y^P\f$; innovation vectors.
   */
  matrix_type J1, J2;

  /**
   * Matrices of sigma points.
   */
  matrix_type X1, X2;

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
   * Create conditional unscented particle filter.
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

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::ConditionalUnscentedParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO3* out) :
    particle_filter_type(m, rng, delta, in, obs, out),
    kalman_filter_type(m, delta, (in == NULL) ? NULL : new IO1(*in),
        (obs == NULL) ? NULL : new IO2(*obs)) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T >= particle_filter_type::getTime());
  assert (relEss >= 0.0 && relEss <= 1.0);

  typedef typename locatable_vector<L,real>::type V3;
  typedef typename locatable_vector<L,int>::type V4;

  /* ukf temps */
  Static<L> theta1(kalman_filter_type::m, theta.size());
  State<L> s1(kalman_filter_type::m);

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* filter */
  init(theta, lws, as);
  int n = 0, r = 0;
  while (particle_filter_type::getTime() < T) {
    prepare(T, theta, s, theta1, s1);
    propose(as, lws);
    particle_filter_type::predict(T, theta, s);
    particle_filter_type::correct(s, lws);
    particle_filter_type::output(n, theta, s, r, lws, as);
    ++n;
    r = particle_filter_type::getTime() < T && resample(theta, s, lws, as,
        resam, relEss);
  }

  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R, class V1>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, const V1 x0, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr,
    R* resam, const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class T1, class V1, class V2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::summarise(
    T1* ll, V1* lls, V2* ess) {
  summarise_pf(this->logWeightsCache, ll, lls, ess);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  particle_filter_type::reset();
  kalman_filter_type::reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L1, bi::Location L2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::prepare(
    const real T, Static<L1>& theta1, State<L1>& s1, Static<L2>& theta2,
    State<L2>& s2) {
  /* pre-condition */
  assert (!haveParameters);

  real tj = obs(T, s2);
  parameters(tj);

  P1 = s1.size();
  P2 = kalman_filter_type::P;

  if (nupdates > 0) {
    muY.resize(W, P1, false);
    SigmaY.resize(W, P1*W, false);
    RY.resize(W, P1*W, false);
    ldetRY.resize(P1, false);
    X1.resize(P1*P2, N2);
    X2.resize(P1*P2, N2);
    J1.resize(W, P1);
    J2.resize(W, P1);

    /* construct and propagate sigma points */
    BOOST_AUTO(Z, temp_matrix<matrix_type>(P1, ND + NC + NR));
    columns(*Z, 0, ND) = s1.get(D_NODE);
    columns(*Z, ND, NC) = s1.get(C_NODE);
    columns(*Z, ND + NC, NR) = s1.get(R_NODE);
    log_columns(columns(*Z, 0, ND), kalman_filter_type::m.getLogs(D_NODE));
    log_columns(columns(*Z, ND, NC), kalman_filter_type::m.getLogs(C_NODE));

    #pragma omp parallel for
    for (int p = 0; p < P1; ++p) {
      set_rows(subrange(X1, p*P2, P2, 0, ND + NC + NR), row(*Z, p));
    }

    s2.resize(P1*P2, false);
    ///@todo Next two lines may not be necessary
    theta2.resize(1);
    theta2 = theta1;
    advanceNoise(tj, X1, X2, theta2, s2);

    /* start on innovations */
    BOOST_AUTO(y, duplicate_vector(row(s2.get(OY_NODE), 0)));
    log_vector(*y, oLogs);
    set_columns(J1, *y);

    /* construct observation densities */
    #pragma omp parallel
    {
      int p;

      /* observation means */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(Y1, subrange(X2, p*P2, P2, N2 - W, W));
        BOOST_AUTO(muY1, column(muY, p));

        gemv(1.0, Y1, Wm, 0.0, muY1, 'T');
      }

      /* observation covariances */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(Y1, subrange(X2, p*P2, P2, N2 - W, W));
        BOOST_AUTO(muY1, column(muY, p));
        BOOST_AUTO(SigmaY1, columns(SigmaY, p*W, W));

        sub_rows(Y1, muY1);
        syrk(Wi, rows(Y1, 1, 2*N1), 0.0, SigmaY1, 'U', 'T');
        syr(Wc0, row(Y1, 0), SigmaY1, 'U');
      }

      /* observation Cholesky factors */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(SigmaY1, columns(SigmaY, p*W, W));
        BOOST_AUTO(RY1, columns(RY, p*W, W));

        chol(SigmaY1, RY1, 'U');
      }

      /* complete and scale innovations */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
        BOOST_AUTO(RY1, columns(RY, p*W, W));
        BOOST_AUTO(muY1, column(muY, p));

        axpy(-1.0, muY1, column(J1, p));
        column(J2, p) = column(J1, p);
        potrs(RY1, columns(J2, p, 1), 'U');
      }
    }

    synchronize();
    delete Z;
    delete y;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, bi::Location L1, bi::Location L2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::prepare(
    const real T, const V1 x0, Static<L1>& theta1, State<L1>& s1,
    Static<L2>& theta2, State<L2>& s) {

}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class V2>
void bi::ConditionalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::propose(
    const V1 as, V2 lws) {
  if (nupdates > 0) {
    BOOST_AUTO(tmp, host_temp_matrix<real>(V, P1));
    BOOST_AUTO(as1, host_map_vector(as));
    V2 lw1(P1), lw2(P1); // weight correction vectors

    SigmaUY.resize(V, P1*W, false);
    muU.resize(V, P1, false);
    RU.resize(V, P1*V, false);
    ldetRU.resize(P1, false);

    /* initial samples */
    BOOST_AUTO(&U, particle_filter_type::rUpdater.buf());
    U.resize(P1, V);
    particle_filter_type::rng.gaussians(vec(U));
    particle_filter_type::rUpdater.setNext(nupdates);
    dot_rows(U, lw1);

    #pragma omp parallel
    {
      int p, w, offset;

      /* noise-observation cross-covariances */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
//        if ((*as1)(p) == p) {
          offset = ND + NC + NR + NR*(nextra - nupdates);
          BOOST_AUTO(Y1, subrange(X2, p*P2 + 1 + offset, V, N2 - W, W));
          BOOST_AUTO(Y2, subrange(X2, p*P2 + 1 + N1 + offset, V, N2 - W, W));
          BOOST_AUTO(U1, subrange(X2, p*P2 + 1 + offset, V, offset, V));
          BOOST_AUTO(U2, subrange(X2, p*P2 + 1 + N1 + offset, V, offset, V));
          BOOST_AUTO(SigmaUY1, columns(SigmaUY, p*W, W));

          gdmm(Wi, diagonal(U1), Y1, 0.0, SigmaUY1);
          gdmm(Wi, diagonal(U2), Y2, 1.0, SigmaUY1);
//        }
      }

      /* compute conditional means */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
//        if ((*as1)(p) == p) {
          BOOST_AUTO(SigmaUY1, columns(SigmaUY, p*W, W));
          BOOST_AUTO(muU1, column(muU, p));

          gemv(1.0, SigmaUY1, column(J2, p), 0.0, muU1);
//        }
      }

      /* compute conditional Cholesky factors */
      #pragma omp for
      for (p = 0; p < P1; ++p) {
//        if ((*as1)(p) == p) {
          BOOST_AUTO(SigmaY1, columns(SigmaY, p*W, W));
          BOOST_AUTO(SigmaUY1, columns(SigmaUY, p*W, W));
          BOOST_AUTO(RU1, columns(RU, p*V, V));
          BOOST_AUTO(RY1, columns(RY, p*W, W));
          BOOST_AUTO(muU1, column(muU, p));

          try {
            /* conditional covariance Cholesky factor */
            trsm(1.0, RY1, SigmaUY1, 'R', 'U');
            ident(RU1);
            for (w = 0; w < W; ++w) {
              ch1dn(RU1, column(SigmaUY1, w), column(*tmp, p));
            }
            ldetRU(p) = log(bi::prod(diagonal(RU1).begin(),
                diagonal(RU1).end(), 1.0));
          } catch (Exception e) {
            ident(RU1);
            muU1.clear();
            ldetRU(p) = 0.0;
          }
//        }
      }

      /* transform samples */
      int q;
      #pragma omp for
      for (p = 0; p < P1; ++p) {
//	     	q = (*as1)(p);
        q = p;
  	    BOOST_AUTO(RU1, columns(RU, q*V, V));
        BOOST_AUTO(muU1, column(muU, q));
        BOOST_AUTO(u1, row(U, q));

        trmv(RU1, u1, 'U');
        axpy(1.0, muU1, u1);
      }
    }

    /* weight correct */
    dot_rows(U, lw2);
    axpy(0.5, lw1, lws);
    axpy(-0.5, lw2, lws);
    axpy(1.0, ldetRU, lws);

    synchronize();
    delete tmp;
    delete as1;
  }
}

#endif
