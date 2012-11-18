/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_AUXILIARYPARTICLEFILTER_HPP
#define BI_METHOD_AUXILIARYPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"

namespace bi {
/**
 * Auxiliary particle filter with lookahead.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class S, class R, class IO1>
class AuxiliaryParticleFilter: public ParticleFilter<B,S,R,IO1> {
public:
  using ParticleFilter<B,S,R,IO1>::resample;

  /**
   * @copydoc ParticleFilter::ParticleFilter()
   */
  AuxiliaryParticleFilter(B& m, S* sim = NULL, R* resam = NULL,
      const real essRel = 1.0, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc ParticleFilter::filter(Random&, const real, const real, cosnt int, State<B,L>&, IO2*)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const real t, const real T, const int K,
      State<B,L>& s, IO2* inInit);

  /**
   * @copydoc ParticleFilter::filter(Random&, const real, const real, const int, const V1, State<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s);

  /**
   * @copydoc ParticleFilter::filter(Random&, const real, const real, const int, const V1, State<B,L>&, M1)
   */
  template<Location L, class V1, class M1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s, M1 X);
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
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param[out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(Random& rng, const real t, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as,
      IO2* inInit);

  /**
   * Initialise, with fixed parameters and starting at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param theta Parameters.
   * @param[out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const real t, const V1 theta, State<B,L>& s, V2 lw1s,
      V2 lw2s, V3 as);

  /**
   * Resample, predict and correct.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[in,out] s State.
   * @param[in,out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * @param[out] as Ancestry after resampling.
   */
  template<bi::Location L, class V1, class V2>
  int step(Random& rng, const real T, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param T Maximum end time for any lookahead.
   * @param[in,out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, const real T, State<B,L>& s, V1 lw1s, V1 lw2s,
      V2 as);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param T Maximum end time for any lookahead.
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, const real T, State<B,L>& s, const int a,
      V1 lw1s, V1 lw2s, V2 as);
  //@}

protected:
  /**
   * Perform lookahead.
   *
   * @param T Maximum end time for lookahead.
   * @param s State.
   * @param[in,out] lw1s On input, current log-weights of particles, on
   * output, stage 1 log-weights.
   */
  template<Location L, class V1>
  void lookahead(Random& rng, const real T, State<B,L>& s, V1 lw1s);
};

/**
 * Factory for creating AuxiliaryParticleFilter objects.
 *
 * @ingroup method
 *
 * @see AuxiliaryParticleFilter
 */
struct AuxiliaryParticleFilterFactory {
  /**
   * Create auxiliary particle filter.
   *
   * @return AuxiliaryParticleFilter object. Caller has ownership.
   *
   * @see AuxiliaryParticleFilter::AuxiliaryParticleFilter()
   */
  template<class B, class S, class R, class IO1>
  static AuxiliaryParticleFilter<B,S,R,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, const real essRel = 1.0, IO1* out = NULL) {
    return new AuxiliaryParticleFilter<B,S,R,IO1>(m, sim, resam, essRel, out);
  }
};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::AuxiliaryParticleFilter<B,S,R,IO1>::AuxiliaryParticleFilter(B& m, S* sim,
    R* resam, const real essRel, IO1* out) :
    ParticleFilter<B,S,R,IO1>(m, sim, resam, essRel, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, State<B,L>& s, IO2* inInit) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  init(rng, t, s, lw1s, lw2s, as, inInit);
  this->output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      r = step(rng, tk, s, lw1s, lw2s, as);
      ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s)
          - 2.0 * bi::log(static_cast<real>(P));
      this->output(n++, s, r, lw2s, as);
    } while (this->getSim()->getTime() < tk);

    ++k;
  } while (k <= K);
  this->term();

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, const V1 theta, State<B,L>& s) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  init(rng, t, theta, s, lw1s, lw2s, as);
  this->output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      r = step(rng, tk, s, lw1s, lw2s, as);
      ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s)
          - 2.0 * bi::log(static_cast<real>(P));
      this->output(n++, s, r, lw2s, as);
    } while (this->getSim()->getTime() < tk);

    ++k;
  } while (k <= K);
  this->term();

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class M1>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, const V1 theta, State<B,L>& s, M1 X) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  init(rng, t, theta, s, lw1s, lw2s, as);
  this->output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : T * k / K;

    /* advance */
    do {
      r = resample(rng, T, s, 0, lw1s, lw2s, as);
      this->predict(rng, T, s);

      /* overwrite first particle with conditioned particle */
      row(s.getDyn(), 0) = column(X, n);

      this->correct(s, lw2s);
      ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s)
          - 2.0 * bi::log(static_cast<real>(P));
      this->output(n++, s, r, lw2s, as);
    } while (this->getSim()->getTime() < tk);

    ++k;
  } while (k <= K);
  this->term();

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class IO2>
void bi::AuxiliaryParticleFilter<B,S,R,IO1>::init(Random& rng, const real t,
    State<B,L>& s, V1 lw1s, V1 lw2s, V2 as, IO2* inInit) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());
  BI_ASSERT(lw2s.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, t, s, lw1s, as, inInit);
  lw2s.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class V3>
void bi::AuxiliaryParticleFilter<B,S,R,IO1>::init(Random& rng, const real t,
    const V1 theta, State<B,L>& s, V2 lw1s, V2 lw2s, V3 as) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());
  BI_ASSERT(lw2s.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, t, theta, s, lw1s, as);
  lw2s.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
int bi::AuxiliaryParticleFilter<B,S,R,IO1>::step(Random& rng, const real T,
    State<B,L>& s, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());

  int r = resample(rng, T, s, lw1s, lw2s, as);
  this->predict(rng, T, s);
  this->correct(s, lw2s);

  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::AuxiliaryParticleFilter<B,S,R,IO1>::resample(Random& rng,
    const real T, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());

  bool r = false;
  this->normalise(lw2s);
  if (this->getSim()->getObs()->hasNext()) {
    const real to = this->getSim()->getObs()->getNextTime();
    lw1s = lw2s;
    if (this->resam != NULL && to > this->getSim()->getTime()) {
      this->lookahead(rng, T, s, lw1s);
      if (this->essRel >= 1.0
          || this->resam->ess(lw1s) <= s.size() * this->essRel) {
        this->resam->resample(rng, lw1s, lw2s, as, s);
        r = true;
      } else {
        lw1s = lw2s;
        seq_elements(as, 0);
      }
    }
  }
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::AuxiliaryParticleFilter<B,S,R,IO1>::resample(Random& rng,
    const real T, State<B,L>& s, const int a, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());
  BI_ASSERT(a >= 0 && a < lw1s.size());

  bool r = false;
  lw1s = lw2s;
  if (this->getSim()->getObs()->hasNext()) {
    const real to = this->getSim()->getObs()->getNextTime();
    if (this->resam != NULL && to > this->getSim()->getTime()) {
      this->lookahead(rng, T, s, lw1s);
      if (this->essRel >= 1.0
          || this->resam->ess(lw1s) <= s.size() * this->essRel) {
        this->resam->resample(rng, a, lw1s, lw2s, as, s);
        r = true;
      } else {
        lw1s = lw2s;
        seq_elements(as, 0);
      }
    }
  }

  this->normalise(lw2s);
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
void bi::AuxiliaryParticleFilter<B,S,R,IO1>::lookahead(Random& rng,
    const real T, State<B,L>& s, V1 lw1s) {
  if (this->getSim()->getObs()->hasNext()) {
    const real to = this->getSim()->getObs()->getNextTime();

    if (to <= T) {
      /* store current state */
      typename loc_temp_matrix<L,real>::type X(s.size(), B::ND + B::NR);
      X = s.getDyn();
      this->mark();

      /* auxiliary lookahead */
      do {
        this->getSim()->lookahead(rng, to, s);
      } while (this->getSim()->getTime() < to);

      /* stage one weights */
      //this->correct(s, lw1s);
      this->m.lookaheadObservationLogDensities(s,
          this->getSim()->getObs()->getMask(), lw1s);

      /* restore previous state */
      s.getDyn() = X;
      this->restore();
    }
  }
}

#endif
