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
 * Auxiliary particle filter with deterministic lookahead.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::ParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class R, class IO1, class IO2, class IO3,
    Location CL = ON_HOST>
class AuxiliaryParticleFilter : public ParticleFilter<B,R,IO1,IO2,IO3,CL> {
public:
  using ParticleFilter<B,R,IO1,IO2,IO3,CL>::resample;

  /**
   * @copydoc ParticleFilter::ParticleFilter()
   */
  AuxiliaryParticleFilter(B& m, R* resam = NULL, const real essRel = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc ParticleFilter::filter(Random&, const real, State<B,L>&, IO4*)
   */
  template<Location L, class IO4>
  real filter(Random& rng, const real T, State<B,L>& s, IO4* inInit);

  /**
   * @copydoc ParticleFilter::filter(Random&, const real, const V1, State<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const real T, const V1 theta0, State<B,L>& s);

  /**
   * @copydoc ParticleFilter::filter(Random&, const real, State<B,:>&, M1, M1)
   */
  template<Location L, class M1>
  real filter(Random& rng, const real T, State<B,L>& s, M1 xd, M1 xr);
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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO4>
  void init(Random& rng, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as, IO4* inInit);

  /**
   * Initialise, with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param theta0 Parameters.
   * @param s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const V1 theta0, State<B,L>& s, V2 lw1s, V2 lw2s, V3 as);

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
  void lookahead(const real T, State<B,L>& s, V1 lw1s);

  /**
   * Cache for stage 1 log-weights.
   */
  Cache2D<real> stage1LogWeightsCache;

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
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
   * Create auxiliary particle filter.
   *
   * @return AuxiliaryParticleFilter object. Caller has ownership.
   *
   * @see AuxiliaryParticleFilter::AuxiliaryParticleFilter()
   */
  template<class B, class R, class IO1, class IO2, class IO3>
  static AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>* create(B& m, R* resam = NULL,
      const real essRel = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>(m, resam, essRel,
        in, obs, out);
  }
};

}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::AuxiliaryParticleFilter(
    B& m, R* resam, const real essRel, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B,R,IO1,IO2,IO3,CL>(m, resam, essRel, in, obs, out) {
  //
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class IO4>
real bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, State<B,L>& s, IO4* inInit) {
  /* pre-conditions */
  assert (T >= this->getTime());

  const int P = s.size();
  int n = 0, r = 0;
  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  real ll = 0.0;
  init(rng, s, lw1s, lw2s, as, inInit);
  while (this->getTime() < T) {
    r = resample(rng, T, s, lw1s, lw2s, as);
    predict(rng, T, s);
    correct(s, lw2s);

    ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s) - 2.0*std::log((real)P);

    output(n, s, r, lw2s, as);
    ++n;
  }
  synchronize();
  this->term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
real bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, const V1 theta0, State<B,L>& s) {
  /* pre-conditions */
  assert (T >= this->getTime());

  const int P = s.size();
  int n = 0, r = 0;
  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  real ll = 0.0;
  init(rng, theta0, s, lw1s, lw2s, as);
  while (this->getTime() < T) {
    r = resample(rng, T, s, lw1s, lw2s, as);
    predict(rng, T, s);
    correct(s, lw2s);

    ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s) - 2.0*std::log((real)P);

    output(n, s, r, lw2s, as);
    ++n;
  }
  synchronize();
  this->term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
real bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, State<B,L>& s, M1 xd, M1 xr) {
  /* pre-condition */
  assert (T >= this->state.t);

  const int P = s.size();
  int n = 0, r = 0, a = 0;

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int>::type as(P);

  real ll = 0.0;
  init(rng, s, lw1s, lw2s, as);
  while (this->state.t < T) {
    r = resample(rng, T, s, a, lw1s, lw2s, as);
    predict(rng, T, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_VAR), 0) = column(xd, n);
    row(s.get(R_VAR), 0) = column(xr, n);

    correct(s, lw2s);

    ll += logsumexp_reduce(lw1s) + logsumexp_reduce(lw2s) - 2.0*std::log(P);

    output(n, s, r, lw2s, as);
    ++n;
  }
  synchronize();
  this->term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class IO4>
void bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::init(Random& rng,
    State<B,L>& s, V1 lw1s, V1 lw2s, V2 as, IO4* inInit) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());
  assert (lw2s.size() == as.size());

  ParticleFilter<B,R,IO1,IO2,IO3,CL>::init(rng, s, lw1s, as, inInit);
  set_elements(lw2s, 0.0);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class V3>
void bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::init(Random& rng,
    const V1 theta0, State<B,L>& s, V2 lw1s, V2 lw2s, V3 as) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());
  assert (lw2s.size() == as.size());

  ParticleFilter<B,R,IO1,IO2,IO3,CL>::init(rng, theta0, s, lw1s, as);
  set_elements(lw2s, 0.0);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
bool bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::resample(Random& rng,
    const real T, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());

  bool r = false;
  this->normalise(lw2s);
  if (this->oyUpdater.hasNext()) {
    const real to = this->oyUpdater.getNextTime();
    lw1s = lw2s;
    if (this->resam != NULL && to > this->state.t) {
      this->lookahead(T, s, lw1s);
      if (this->essRel >= 1.0 || this->resam->ess(lw1s) <= s.size()*this->essRel) {
        this->resam->resample(rng, lw1s, lw2s, as, s);
        r = true;
      } else {
        lw1s = lw2s;
        seq_elements(as, 0);
      }

      /* post-condition */
      assert (this->sim.getTime() == this->getTime());
    }
  }
  return r;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
bool bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::resample(Random& rng,
    const real T, State<B,L>& s, const int a, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-condition */
  assert (lw1s.size() == lw2s.size());
  assert (a >= 0 && a < lw1s.size());

  bool r = false;
  lw1s = lw2s;
  if (this->oyUpdater.hasNext()) {
    const real to = this->oyUpdater.getNextTime();
    if (this->resam != NULL && to > this->state.t) {
      this->lookahead(T, s, lw1s);
      if (this->essRel >= 1.0 || this->resam->ess(lw1s) <= s.size()*this->essRel) {
        this->resam->resample(rng, a, lw1s, lw2s, as, s);
        r = true;
      } else {
        lw1s = lw2s;
        seq_elements(as, 0);
      }

      /* post-condition */
      assert (this->sim.getTime() == this->getTime());
    }
  }

  this->normalise(lw2s);
  return r;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
void bi::AuxiliaryParticleFilter<B,R,IO1,IO2,IO3,CL>::lookahead(const real T,
    State<B,L>& s, V1 lw1s) {
  if (this->oyUpdater.hasNext()) {
    const real to = this->oyUpdater.getNextTime();
    if (to <= T) {
      typename loc_temp_matrix<L,real>::type X(s.size(), ND + NR);
      real delta = B::getDelta();
      int nupdates = lt_steps(to, delta) - lt_steps(this->state.t, delta);
      real to;

      /* store current state */
      X = s.getDyn();
      this->mark();

      /* auxiliary simulation */
      while (this->state.t < to) {
        this->sim.lookahead(to, s);
        this->state.t = this->sim.getTime();
      }

      /* stage one weights */
      //this->correct(s, lw1s);
      this->oyUpdater.update(s);
      this->m.lookaheadObservationLogDensities(s, this->oyUpdater.getMask(),
          lw1s);

      /* restore previous state */
      s.getDyn() = X;
      this->restore();
    }
  }

  /* post-condition */
  assert (this->sim.getTime() == this->state.t);
}

#endif
