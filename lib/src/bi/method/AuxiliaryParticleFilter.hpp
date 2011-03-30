/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1246 $
 * $Date: 2011-01-31 16:23:46 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_METHOD_AUXILIARYPARTICLEFILTER_HPP
#define BI_METHOD_AUXILIARYPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"

namespace bi {
/**
 * Auxiliary particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::AuxiliaryParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class AuxiliaryParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL> {
public:
  using ParticleFilter<B,IO1,IO2,IO3,CL>::resample;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  AuxiliaryParticleFilter(B& m, Random& rng, IO1* in = NULL,
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

  /**
   * @copydoc #concept::Filter::summarise()
   *
   * Uses marginal likelihood estimator of @ref Pitt2002 "Pitt (2002)".
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
   * @param s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   */
  template<Location L, class V1, class V2, class R>
  void resample(State<L>& s, V1& lw1s, V1& lw2s, V2& as, R* resam = NULL);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   */
  template<Location L, class V1, class V2, class R>
  void resample(State<L>& s, const int a, V1& lw1s, V1& lw2s, V2& as,
      R* resam = NULL);

  /**
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param k Time index.
   * @param s State.
   * @param r 1 if resampling was performed before moving to this time, 0
   * otherwise.
   * @param lw1s Stage 1 log-weights.
   * @param lw2s Stage 2 log-weights.
   * @param Ancestry.
   */
  template<Location L, class V1, class V2>
  void output(const int k, const State<L>& s, const int r, const V1& lw1s,
      const V1& lw2s, const V2& as);
  //@}

private:
  /**
   * Perform lookahead.
   *
   * @param s State.
   * @param[in,out] lw1s On input, current log-weights of particles, on
   * output, stage 1 log-weights.
   */
  template<Location L, class V1>
  void lookahead(State<L>& s, V1& lw1s);
};

/**
 * Factory for creating AuxiliaryParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see AuxiliaryParticleFilter
 */
template<Location CL = ON_HOST>
struct AuxiliaryParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return AuxiliaryParticleFilter object. Caller has ownership.
   *
   * @see AuxiliaryParticleFilter::AuxiliaryParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>* create(B& m,
      Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs,
        out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::AuxiliaryParticleFilter(
    B& m, Random& rng, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs, out) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class R>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam) {
  /* pre-condition */
  assert (T > this->state.t);

  int n = 0, r = 0;

  BOOST_AUTO(lw1s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(lw2s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  init(theta, *lw1s, *lw2s, *as);
  resample(s, *lw1s, *lw2s, *as, resam);
  while (this->state.t < T) {
    predict(T, s);
    correct(s, *lw2s);
    output(n, s, r, *lw1s, *lw2s, *as);
    ++n;
    r = this->state.t < T; // no need to resample at last time
    if (r) {
      resample(s, *lw2s, *as, resam);
      resample(s, *lw1s, *lw2s, *as, resam);
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
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam) {
  /* pre-condition */
  assert (T > this->state.t);
  assert (this->out != NULL);

  int n = 0, r = 0, a = 0;

  BOOST_AUTO(lw1s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(lw2s, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  init(theta, *lw1s, *lw2s, *as);
  resample(s, *lw1s, *lw2s, *as, resam);
  while (this->state.t < T) {
    predict(T, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_NODE), 0) = column(xd, n);
    row(s.get(C_NODE), 0) = column(xc, n);
    row(s.get(R_NODE), 0) = column(xr, n);

    correct(s, *lw2s);
    output(n, s, r, *lw1s, *lw2s, *as);
    ++n;
    r = this->state.t < T; // no need to resample at last time
    if (r) {
      this->resample(s, a, *lw2s, *as, resam);
      resample(s, a, *lw1s, *lw2s, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lw1s;
  delete lw2s;
  delete as;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class T1, class V1, class V2>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::summarise(T1* ll, V1* lls, V2* ess) {
  /* pre-condition */
  BI_ERROR(this->out != NULL,
      "Cannot summarise AuxiliaryParticleFilter without output");

  const int P = this->out->size1();
  const int T = this->out->size2();

  BOOST_AUTO(lw1s1, host_temp_vector<real>(P));
  BOOST_AUTO(lw2s1, host_temp_vector<real>(P));
  BOOST_AUTO(ess1, host_temp_vector<real>(T));
  BOOST_AUTO(lls1, host_temp_vector<real>(T));
  BOOST_AUTO(lls2, host_temp_vector<real>(T));
  real ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real sum1, sum2, sum3;
  for (n = 0; n < T; ++n) {
    this->out->readStage1LogWeights(n, *lw1s1);
    this->out->readLogWeights(n, *lw2s1);

    //bi::sort(lw1s1->begin(), lw1s1->end());
    //bi::sort(lw2s1->begin(), lw2s1->end());

    sum1 = sum_exp(lw1s1->begin(), lw1s1->end(), 0.0);
    sum2 = sum_exp(lw2s1->begin(), lw2s1->end(), 0.0);
    sum3 = sum_exp_square(lw2s1->begin(), lw2s1->end(), 0.0);

    (*lls1)(n) = log(sum1) + log(sum2) - 2*log(P);
    (*ess1)(n) = (sum2*sum2) / sum3;
  }

  /* compute marginal log-likelihood */
  *lls2 = *lls1;
  //bi::sort(lls2->begin(), lls2->end());
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

  delete lw1s1;
  delete lw2s1;
  delete ess1;
  delete lls1;
  delete lls2;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::init(Static<L>& theta,
    V1& lw1s, V1& lw2s, V2& as) {
  /* pre-condition */
  assert (lw2s.size() == as.size());

  ParticleFilter<B,IO1,IO2,IO3,CL>::init(theta, lw1s, as);

  bi::fill(lw2s.begin(), lw2s.end(), 0.0);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::resample(State<L>& s,
    V1& lw1s, V1& lw2s, V2& as, R* resam) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;

  const real to = this->oyUpdater.getTime();
  if (resam != NULL && to >= this->state.t) {
    lw1s = lw2s;
    this->lookahead(s, lw1s);
    resam->resample(lw1s, lw2s, as);
    this->copy(s, as, resam);

    /* post-condition */
    assert (this->sim.getTime() == this->getTime());
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::resample(State<L>& s,
    const int a, V1& lw1s, V1& lw2s, V2& as, R* resam) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;

  const real to = this->oyUpdater.getTime();
  if (resam != NULL && to >= this->state.t) {
    lw1s = lw2s;
    this->lookahead(s, lw1s);
    resam->resample(a, lw1s, lw2s, as);
    this->copy(s, as, resam);

    /* post-condition */
    assert (this->sim.getTime() == this->getTime());
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::output(const int k,
    const State<L>& s, const int r, const V1& lw1s, const V1& lw2s,
    const V2& as) {
  ParticleFilter<B,IO1,IO2,IO3,CL>::output(k, s, r, lw2s, as);

  if (this->haveOut) {
    this->out->writeStage1LogWeights(k, lw1s);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
void bi::AuxiliaryParticleFilter<B,IO1,IO2,IO3,CL>::lookahead(State<L>& s,
    V1& lw1s) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;
  typedef typename locatable_temp_matrix<L,real>::type temp_matrix_type;

  const int P = s.size();
  const int ND = this->m.getNetSize(D_NODE);
  const int NC = this->m.getNetSize(C_NODE);

  temp_matrix_type X(P, ND + NC);
  const real to = this->oyUpdater.getTime();

  /* store current state */
  columns(X, 0, ND) = s.get(D_NODE);
  columns(X, ND, NC) = s.get(C_NODE);
  this->mark();

  /* auxiliary simulation forward */
  //s.get(R_NODE).clear(); // no-noise lookahead
  //this->rUpdater.skipNext();
  this->predict(to, s);
  this->correct(s, lw1s);

  /* restore previous state */
  s.get(D_NODE) = columns(X, 0, ND);
  s.get(C_NODE) = columns(X, ND, NC);
  this->restore();
}


#endif
