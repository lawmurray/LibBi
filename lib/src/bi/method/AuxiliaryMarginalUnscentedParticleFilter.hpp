/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1780 $
 * $Date: 2011-07-21 10:31:47 +0800 (Thu, 21 Jul 2011) $
 */
#ifndef BI_METHOD_AUXILIARYMARGINALUNSCENTEDPARTICLEFILTER_HPP
#define BI_METHOD_AUXILIARYMARGINALUNSCENTEDPARTICLEFILTER_HPP

#include "MarginalUnscentedParticleFilter.hpp"

namespace bi {
/**
 * Auxiliary marginal unscented particle filter with deterministic lookahead
 * using mean from UKF.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::AuxiliaryMarginalUnscentedParticleFilterBuffer type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST,
    StaticHandling SH = STATIC_SHARED>
class AuxiliaryMarginalUnscentedParticleFilter :
    public MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH> {
public:
  /**
   * Particle filter type.
   */
  typedef typename MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::particle_filter_type particle_filter_type;

  /**
   * Kalman filter type.
   */
  typedef typename MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::kalman_filter_type kalman_filter_type;

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
  AuxiliaryMarginalUnscentedParticleFilter(B& m, Random& rng,
      const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc MarginalUnscentedParticleFilter::filter()
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL,
      const real relEss = 1.0);

  /**
   * @copydoc MarginalUnscentedParticleFilter::filter()
   */
  template<Location L, class R, class V1>
  void filter(const real T, const V1 x0, Static<L>& theta, State<L>& s,
      R* resam = NULL, const real relEss = 1.0);

  /**
   * @copydoc MarginalUnscentedParticleFilter::filter()
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL, const real relEss = 1.0);

  /**
   * @copydoc summarise_apf()
   */
  template<class T1, class V1, class V2>
  void summarise(T1* ll, V1* lls, V2* ess);
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
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param theta Static state.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2>
  void init(Static<L>& theta, V1& lw1s, V1& lw2s, V2& as);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param theta Static state.
   * @param s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2, class R>
  bool resample(Static<L>& theta, State<L>& s, V1& lw1s, V1& lw2s, V2& as,
      R* resam = NULL, const real relEss = 1.0);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param theta Static state.
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2, class R>
  bool resample(Static<L>& theta, State<L>& s, const int a, V1& lw1s,
      V1& lw2s, V2& as, R* resam = NULL, const real relEss = 1.0);

  /**
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param k Time index.
   * @param theta Static state.
   * @param s State.
   * @param r 1 if resampling was performed before moving to this time, 0
   * otherwise.
   * @param lw1s Stage 1 log-weights.
   * @param lw2s Stage 2 log-weights.
   * @param Ancestry.
   */
  template<Location L, class V1, class V2>
  void output(const int k, const Static<L>& theta, const State<L>& s,
      const int r, const V1& lw1s, const V1& lw2s, const V2& as);

  /**
   * Flush output caches to file.
   */
  void flush();
  //@}

protected:
  /**
   * Perform lookahead.
   *
   * @param theta Static state.
   * @param s State.
   * @param[in,out] lw1s On input, current log-weights of particles, on
   * output, stage 1 log-weights.
   */
  template<Location L, class V1>
  void lookahead(Static<L>& theta, State<L>& s, V1& lw1s);

  /**
   * Cache for stage 1 log-weights.
   */
  Cache2D<real> stage1LogWeightsCache;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
};

/**
 * Factory for creating AuxiliaryMarginalUnscentedParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see AuxiliaryMarginalUnscentedParticleFilter
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct AuxiliaryMarginalUnscentedParticleFilterFactory {
  /**
   * Create auxiliary marginal unscented particle filter.
   *
   * @return AuxiliaryMarginalUnscentedParticleFilter object. Caller has
   * ownership.
   *
   * @see AuxiliaryMarginalUnscentedParticleFilter::AuxiliaryMarginalUnscentedParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(
      B& m, Random& rng, const real delta = 1.0, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL) {
    return new AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>(
        m, rng, delta, in, obs, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::AuxiliaryMarginalUnscentedParticleFilter(
    B& m, Random& rng, const real delta, IO1* in, IO2* obs, IO3* out) :
    MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in,
        obs, out) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T > this->getTime());
  assert (relEss >= 0.0 && relEss <= 1.0);

  typedef typename locatable_vector<ON_HOST,real>::type V2;
  typedef typename locatable_matrix<ON_HOST,real>::type M2;
  typedef typename locatable_vector<L,real>::type V3;
  typedef typename locatable_vector<L,int>::type V4;

  /* ukf temps */
  Static<ON_HOST> theta1(particle_filter_type::m, theta.size());
  State<ON_HOST> s1(particle_filter_type::m);

  /* pf temps */
  V3 lw1s(s.size()), lw2s(s.size());
  V4 as(s.size());

  /* filter */
  init(theta, lw1s, lw2s, as);
  #ifndef USE_CPU
  #pragma omp parallel sections
  #endif
  {
    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      if (!particle_filter_type::haveParameters) {
        theta1 = theta;
      }
      this->prepare(T, theta1, s1);
    }

    #ifndef USE_CPU
    #pragma omp section
    #endif
    {
      int n = 0, r = 0;
      while (particle_filter_type::getTime() < T) {
        r = resample(theta, s, lw1s, lw2s, as, resam, relEss);
        propose(lw2s);
        particle_filter_type::predict(T, theta, s);
        correct(s, lw2s);
        output(n, theta, s, r, lw1s, lw2s, as);
        ++n;
      }
    }
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class R, class V1>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, const V1 x0, Static<L>& theta, State<L>& s, R* resam,
    const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(
    const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr,
    R* resam, const real relEss) {
  assert (false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class T1, class V1, class V2>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::summarise(
    T1* ll, V1* lls, V2* ess) {
  summarise_apf(stage1LogWeightsCache, this->logWeightsCache, ll, lls, ess);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class V2>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(
    Static<L>& theta, V1& lw1s, V1& lw2s, V2& as) {
  /* pre-condition */
  assert (lw2s.size() == as.size());

  MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(theta, lw1s,
      as);
  bi::fill(lw2s.begin(), lw2s.end(), 0.0);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class R>
bool bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::resample(
    Static<L>& theta, State<L>& s, V1& lw1s, V1& lw2s, V2& as, R* resam,
    const real relEss) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());

  bool r = false;
  this->normalise(lw2s);
  if (particle_filter_type::oyUpdater.hasNext()) {
    const real to = particle_filter_type::oyUpdater.getNextTime();
    lw1s = lw2s;
    if (resam != NULL && to > this->getTime()) {
      this->lookahead(theta, s, lw1s);
      if (relEss >= 1.0 || ess(lw1s) <= s.size()*relEss) {
        resam->resample(lw1s, lw2s, as, theta, s);
        r = true;
      } else {
        lw1s = lw2s;
      }
    }
  }
  return r;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class R>
bool bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::resample(
    Static<L>& theta, State<L>& s, const int a, V1& lw1s, V1& lw2s, V2& as,
    R* resam, const real relEss) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());
  assert (a >= 0 && a < lw1s.size());

  bool r = false;
  this->normalise(lw2s);
  if (this->oyUpdater.hasNext()) {
    const real to = this->oyUpdater.getNextTime();
    lw1s = lw2s;
    if (resam != NULL && to > this->state.t) {
      this->lookahead(theta, s, lw1s);
      if (relEss >= 1.0 || ess(lw1s) <= s.size()*relEss) {
        resam->resample(a, lw1s, lw2s, as, theta, s);
        r = true;
      } else {
        lw1s = lw2s;
      }

      /* post-condition */
      assert (this->sim.getTime() == this->getTime());
    }
  }
  return r;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class V2>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::output(
    const int k, const Static<L>& theta, const State<L>& s, const int r,
    const V1& lw1s, const V1& lw2s, const V2& as) {
  MarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::output(k, theta, s,
      r, lw2s, as);

  if (particle_filter_type::haveOut) {
    stage1LogWeightsCache.put(k, lw1s);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::flush() {
  ParticleFilter<B,IO1,IO2,IO3,CL,SH>::flush();

  int p;
  if (particle_filter_type::haveOut) {
    assert (stage1LogWeightsCache.isValid());
    for (p = 0; p < stage1LogWeightsCache.size(); ++p) {
      this->getOutput()->writeStage1LogWeights(p,
          stage1LogWeightsCache.get(p));
    }
    stage1LogWeightsCache.clean();
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::AuxiliaryMarginalUnscentedParticleFilter<B,IO1,IO2,IO3,CL,SH>::lookahead(
    Static<L>& theta, State<L>& s, V1& lw1s) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;
  typedef typename locatable_temp_matrix<L,real>::type temp_matrix_type;

  #ifndef USE_CPU
  /* ensure prepare() is actually ahead... */
  while (k1 >= k2) {
    #pragma omp flush(k2)
  }
  #endif

  const int P = s.size();

  if (particle_filter_type::oyUpdater.hasNext()) {
    const real to = particle_filter_type::oyUpdater.getNextTime();
    temp_matrix_type X(P, ND + NC + NR);
    real delta = particle_filter_type::sim.getDelta();
    int nupdates = lt_steps(to, delta) - lt_steps(this->getTime(), delta);

    /* store current state */
    columns(X, 0, ND) = s.get(D_NODE);
    columns(X, ND, NC) = s.get(C_NODE);
    columns(X, ND + NC, NR) = s.get(R_NODE);
    particle_filter_type::mark();

    /* auxiliary simulation forward */
    BOOST_AUTO(&U, particle_filter_type::rUpdater.buf());
    U.resize(s.size(), this->mu[this->k1]->size());
    set_rows(U, *this->mu[this->k1]);
    particle_filter_type::rUpdater.setNext(nupdates);
    this->predict(to, theta, s);
    this->correct(s, lw1s);

    /* restore previous state */
    s.get(D_NODE) = columns(X, 0, ND);
    s.get(C_NODE) = columns(X, ND, NC);
    s.get(R_NODE) = columns(X, ND + NC, NR);
    particle_filter_type::restore();
  }
}

#endif
