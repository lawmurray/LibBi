/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MarginalUnscentedParticleFilter_HPP
#define BI_METHOD_MarginalUnscentedParticleFilter_HPP

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
class MarginalUnscentedParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL,SH>,
    protected UnscentedKalmanFilter<B,IO1,IO2,IO3,ON_HOST,SH> {
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
  MarginalUnscentedParticleFilter(B& m, Random& rng, const real delta = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~MarginalUnscentedParticleFilter();

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
   * Lookahead using unscented Kalman filter and use to construct proposal
   * distributions for particle filter.
   *
   * @tparam L Location.
   *
   * @param T Time to which to predict.
   * @param[out] theta Static state for unscented Kalman filter.
   * @param[out] s State for unscented Kalman filter.
   *
   * Note that @p theta and @p s should be different to the analogous
   * arguments provided to other calls, or the particle filter's state
   * will be lost!
   */
  template<Location L>
  void prepare(const real T, Static<L>& theta, State<L>& s);

  /**
   * Lookahead using unscented Kalman filter from fixed starting state
   * and use to construct proposal distributions for particle filter.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param T Time to which to predict.
   * @param x0 Starting state.
   * @param[out] theta Static state for unscented Kalman filter.
   * @param[out] s State for unscented Kalman filter.
   *
   * Note that @p theta and @p s should be different to the analogous
   * arguments provided to other calls, or the particle filter's state
   * will be lost!
   */
  template<Location L, class V1>
  void prepare(const real T, const V1 x0, Static<L>& theta, State<L>& s);

  /**
   * Propose stochastic terms for next particle filter prediction.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] lws Log-weights.
   */
  template<class V1>
  void propose(V1 lws);
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

  /**
   * \f$\hat{\boldsymbol{\mu}}_1,\ldots,\hat{\boldsymbol{\mu}}_T\f$; noise
   * term proposal mean from UKF.
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
 * Factory for creating MarginalUnscentedParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see MarginalUnscentedParticleFilter
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct MarginalUnscentedParticleFilterFactory {
  /**
   * Create marginal unscented particle filter.
   *
   * @return MarginalUnscentedParticleFilter object. Caller has ownership.
   *
   * @see MarginalUnscentedParticleFilter::MarginalUnscentedParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(B& m,
      Random& rng, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng,
        delta, in, obs, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::MarginalUnscentedParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO3* out) :
    particle_filter_type(m, rng, delta, in, obs, out),
    kalman_filter_type(m, delta, (in == NULL) ? NULL : new IO1(*in),
        (obs == NULL) ? NULL : new IO2(*obs)),
    mu(out->size2(), NULL), U(out->size2(), NULL), detU(out->size2(), 1.0) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::~MarginalUnscentedParticleFilter() {
  BOOST_AUTO(iter1, mu.begin());
  for (; iter1 != mu.end(); ++iter1) {
    delete *iter1;
  }
  BOOST_AUTO(iter2, U.begin());
  for (; iter2 != U.end(); ++iter2) {
    delete *iter2;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
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

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* filter */
  init(theta, lws, as);
  #ifndef USE_CPU
  #pragma omp parallel sections
  #endif
  {
    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      if (!haveParameters) {
        theta1 = theta;
      }
      prepare(T, theta1, s1);
    }

    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      int n = 0, r = 0;
      while (particle_filter_type::getTime() < T) {
        propose(lws);
        predict(T, theta, s);
        correct(s, lws);
        output(n, theta, s, r, lws, as);
        ++n;
        r = particle_filter_type::state.t < T && resample(theta, s, lws, as,
            resam, relEss);
      }
    }
  }

  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R, class V1>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, const V1 x0, Static<L>& theta, State<L>& s, R* resam,
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

  /* pf temps */
  V3 lws(s.size());
  V4 as(s.size());

  /* initialise pf from fixed starting state */
  set_rows(s.get(D_NODE), subrange(x0, 0, ND));
  set_rows(s.get(C_NODE), subrange(x0, ND, NC));
  set_rows(theta.get(P_NODE), subrange(x0, ND + NC, NP));

  /* filter */
  init(theta, lws, as);
  #ifndef USE_CPU
  #pragma omp parallel sections
  #endif
  {
    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      if (!haveParameters) {
        theta1 = theta;
      }
      prepare(T, x0, theta1, s1);
    }

    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      int n = 0, r = 0;
      while (getTime() < T) {
        propose(lws);
        predict(T, theta, s);
        correct(s, lws);
        output(n, theta, s, r, lws, as);
        ++n;
        r = getTime() < T && resample(theta, s, lws, as, resam, relEss);
      }
    }
  }

  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr,
    R* resam, const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  particle_filter_type::reset();
  kalman_filter_type::reset();
  k1 = 0;
  k2 = 0;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::prepare(
    const real T, Static<L>& theta, State<L>& s) {
  typedef typename locatable_temp_vector<L,real>::type V1;
  typedef typename locatable_temp_matrix<L,real>::type M1;

  ExpGaussianPdf<V1,M1> corrected(M);
  ExpGaussianPdf<V1,M1> uncorrected(M);
  ExpGaussianPdf<V1,M1> rcorrected(0);
  ExpGaussianPdf<V1,M1> runcorrected(0);
  ExpGaussianPdf<V1,M1> observed(0);
  M1 SigmaXX(M, M), SigmaXY(M, 0), SigmaRY(M, 0);
  M1 X1, X2, Sigma;
  V1 mu;

  real tj;
  kalman_filter_type::init(theta, corrected);
  while (kalman_filter_type::getTime() < T) {
    tj = obs(T, s);

    kalman_filter_type::parameters(tj);
    kalman_filter_type::resize(s, theta, observed, SigmaXY, X1, X2, mu,
        Sigma);
    kalman_filter_type::resizeNoise(runcorrected, rcorrected, SigmaRY);
    kalman_filter_type::sigmas(corrected, X1);
    kalman_filter_type::advance(tj, X1, X2, theta, s);
    kalman_filter_type::mean(X2, mu);
    kalman_filter_type::cov(X2, mu, Sigma);
    kalman_filter_type::predict(mu, Sigma, uncorrected, SigmaXY);
    kalman_filter_type::observe(mu, Sigma, observed);
    kalman_filter_type::predictNoise(mu, Sigma, runcorrected, SigmaRY);
    kalman_filter_type::correct(uncorrected, SigmaXY, s, observed,
        corrected);
    kalman_filter_type::correctNoise(runcorrected, SigmaRY, s, observed,
        rcorrected);

    if (this->mu[k2] == NULL) {
      this->mu[k2] = new vector_type(rcorrected.size());
    }
    if (this->U[k2] == NULL) {
      this->U[k2] = new matrix_type(rcorrected.size(), rcorrected.size());
    }
    *this->mu[k2] = rcorrected.mean();
    *this->U[k2] = rcorrected.std();
    this->detU[k2] = sqrt(rcorrected.det());

    ++k2;
    #ifndef USE_CPU
    #pragma omp flush(k2)
    #endif
  }
  synchronize();
  kalman_filter_type::term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::prepare(
    const real T, const V1 x0, Static<L>& theta, State<L>& s) {

}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1>
void bi::MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::propose(
    V1 lws) {
  #ifndef USE_CPU
  /* ensure prepare() is actually ahead... */
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
    thrust::transform(lws.begin(), lws.end(), lws.begin(),
        add_constant_functor<real>(log(detU)));
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

#endif
