/**
 * @file
 *
 * @author
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_ADAPTIVENPARTICLEFILTER_HPP
#define BI_METHOD_ADAPTIVENPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "../stopper/Stopper.hpp"

namespace bi {
/**
 * Adaptive N Particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R #concept::Resampler type.
 * @tparam S2 #concept::Stopper type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class S, class R, class S2, class IO1>
class AdaptiveNParticleFilter: public ParticleFilter<B,S,R,IO1> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param stopper Stopping criterion for adapting number of particles.
   * @param blockSize Number of particles to propagate per block.
   * @param out Output.
   */
  AdaptiveNParticleFilter(B& m, S* sim = NULL, R* resam = NULL,
      const real essRel = 1.0, S2* stopper = NULL, const int blockSize = 128,
      IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * %Filter forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO2>
  real filter(Random& rng, const real t, const real T, const int K,
      State<B,L>& s, IO2* inInit);

  template<Location L, class V1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s);

  template<Location L, class V1, class M1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s, M1 X);

  template<Location L, class V1, class M1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta_1, State<B,L>& s_1, M1 X, const V1 theta_2,
      State<B,L>& s_2, real& ll1, real& ll2);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  template<bi::Location L, class V1, class V2, class M1>
  int step(Random& rng, const real T, const int totalObs, State<B,L>& s,
      V1& lws, V2& as, int n, const bool conditional, M1* X);

  template<bi::Location L, class V1, class V2>
  int step(Random& rng, const real T, const int totalObs, State<B,L>& s,
      V1& lws, V2& as, int n);

  template<bi::Location L, class V1, class V2, class M1>
  int step(Random& rng, const real T, const int totalObs, State<B,L>& s_1,
      M1 X, State<B,L>& s_2, V1& lws_1, V2& as_1_in, V1& lws_2, V2& as_2_in,
      int n);
  //@}

protected:
  template<class V1, class V2, class V3, class V4>
  bool resample(Random& rng, V1 lws, V2 as, bool sorted, V3 slws, V4 ps,
      V3 Ws);

  template<class V1, class V2, class V3, class V4>
  bool resample(Random& rng, int a, V1 lws, V2 as, bool sorted, V3 slws,
      V4 ps, V3 Ws);

  template<bi::Location L, class M1>
  real filter_impl(Random& rng, const real T, const int K, State<B,L>& s,
      const bool conditional, M1* X);

  template<bi::Location L, class M1>
  real filter_impl(Random& rng, const real T, const int K, State<B,L>& s_1,
      M1 X, State<B,L>& s_2, real& ll1, real& ll2);

  template<bi::Location L, class V1, class V2, class M1>
  int step_impl(Random& rng, const real T, const int totalObs, State<B,L>& s,
      V1& lws, V2& as, int n, const bool conditional, M1* X);

  /**
   * Compute maximum particle weight at current time.
   *
   * @tparam L Location.
   *
   * @param s State.
   *
   * @return Maximum weight.
   */
  template<Location L>
  real getMaxLogWeight(State<B,L>& s);

  S2* stopper;

  const int blockSize;
};

/**
 * Factory for creating AdaptiveNParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see AdaptiveNParticleFilter
 */
struct AdaptiveNParticleFilterFactory {
  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptiveNParticleFilter object. Caller has ownership.
   *
   * @see AdaptiveNParticleFilter::AdaptiveNParticleFilter()
   */
  template<class B, class S, class R, class S2, class IO1>
  static AdaptiveNParticleFilter<B,S,R,S2,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, const real essRel = 1.0, S2* stopper = NULL,
      const int blockSize = 128, IO1* out = NULL) {
    return new AdaptiveNParticleFilter<B,S,R,S2,IO1>(m, sim, resam, essRel,
        stopper, blockSize, out);
  }

  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptiveNParticleFilter object. Caller has ownership.
   *
   * @see AdaptiveNParticleFilter::AdaptiveNParticleFilter()
   */
  template<class B, class S, class R, class S2>
  static AdaptiveNParticleFilter<B,S,R,S2,ParticleFilterCache<> >* create(B& m, S* sim = NULL,
      R* resam = NULL, const real essRel = 1.0, S2* stopper = NULL,
      const int blockSize = 128) {
    return new AdaptiveNParticleFilter<B,S,R,S2,ParticleFilterCache<> >(m, sim, resam, essRel,
        stopper, blockSize);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class S2, class IO1>
bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::AdaptiveNParticleFilter(B& m,
    S* sim, R* resam, const real essRel, S2* stopper, const int blockSize,
    IO1* out) :
    ParticleFilter<B,S,R,IO1>(m, sim, resam, essRel, out), stopper(stopper), blockSize(
        blockSize) {
  //
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class IO2>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const real t, const real T, const int K, State<B,L>& s, IO2* inInit) {
  const int P = s.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  this->init(rng, t, s, lws, as, inInit);

  return filter_impl(rng, T, K, s, false, (host_matrix<real>*)NULL);

}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const real t, const real T, const int K, const V1 theta, State<B,L>& s) {
  const int P = s.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  this->init(rng, t, theta, s, lws, as);

  return filter_impl(rng, T, K, s, false, (host_matrix<real>*)NULL);

}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class M1>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const real t, const real T, const int K, const V1 theta, State<B,L>& s,
    M1 X) {

  const int P = s.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);
  typename loc_temp_vector<L,real>::type lws2(P);

  this->init(rng, t, theta, s, lws, as);

  return filter_impl(rng, T, K, s, true, &X);
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class M1>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const real t, const real T, const int K, const V1 theta_1,
    State<B,L>& s_1, M1 X, const V1 theta_2, State<B,L>& s_2, real& ll1,
    real& ll2) {
  const int P = s_1.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  this->init(rng, t, theta_1, s_1, lws, as);
  this->init(rng, t, theta_2, s_2, lws, as);

  return filter_impl(rng, T, K, s_1, X, s_2, ll1, ll2);
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class V2, class M1>
int bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::step(Random& rng, const real T,
    const int totalObs, State<B,L>& s, V1& lws, V2& as, int n,
    const bool conditional, M1* X) {
  return step_impl(rng, T, totalObs, s, lws, as, n, conditional, X);
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class V2>
int bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::step(Random& rng, const real T,
    const int totalObs, State<B,L>& s, V1& lws, V2& as, int n) {
  return step_impl(rng, T, totalObs, s, lws, as, n, false,
      (host_matrix<real>*)NULL, (host_matrix<real>*)NULL);
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class V2, class M1>
int bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::step(Random& rng, const real T,
    const int totalObs, State<B,L>& s_1, M1 X, State<B,L>& s_2, V1& lws_1,
    V2& as_1_in, V1& lws_2, V2& as_2_in, int n) {

  const int P = s_1.size();
  int r_1 = 0, r_2 = 0;

  int maxParticles = stopper->getMaxParticles();
  typename loc_temp_vector<L,int>::type as_1_base(bi::max(maxParticles, P));
  typename loc_temp_vector<L,real>::type lws2_1_base(
      bi::max(maxParticles, P));

  typename loc_temp_vector<L,int>::type as_2_base(bi::max(maxParticles, P));
  typename loc_temp_vector<L,real>::type lws2_2_base(
      bi::max(maxParticles, P));

  int currentRange = P;

  real maxlw = -1.0 / 0.0;

  // inside loop

  int block = 0;
  int lastSize = s_1.size();
  typename loc_temp_matrix<L,real>::type xvars_1(s_1.size(),
      s_1.getDyn().size2());
  typename loc_temp_matrix<L,real>::type xvars_2(s_2.size(),
      s_2.getDyn().size2());
  xvars_1 = s_1.getDyn();
  xvars_2 = s_2.getDyn();

  typename sim_temp_vector<V1>::type slws_1(P);
  typename sim_temp_vector<V1>::type Ws_1(P);
  typename sim_temp_vector<V2>::type ps_1(P);
  typename sim_temp_vector<V1>::type slws_2(P);
  typename sim_temp_vector<V1>::type Ws_2(P);
  typename sim_temp_vector<V2>::type ps_2(P);

  bool finished = false;
  if (n == 0) {
    this->mark();
    this->predict(rng, T, s_1);
    /* overwrite first particle with conditioned particle */
    row(s_1.getDyn(), 0) = column(X, n);
    this->correct(s_1, lws_1);

    this->restore();
    this->predict(rng, T, s_2);
    this->correct(s_2, lws_2);
  } else {
    currentRange = blockSize;
    this->mark();
    while (!finished) {
      this->top();
      BOOST_AUTO(as_1, subrange(as_1_base, block * blockSize, blockSize));
      BOOST_AUTO(lws2_1, subrange(lws2_1_base, 0, currentRange));
      BOOST_AUTO(as_2, subrange(as_2_base, block * blockSize, blockSize));
      BOOST_AUTO(lws2_2, subrange(lws2_2_base, 0, currentRange));
      if (block == 0) {
        r_1 = resample(rng, 0, lws_1, as_1, false, slws_1, ps_1, Ws_1);
      } else {
        r_1 = resample(rng, lws_1, as_1, true, slws_1, ps_1, Ws_1);
      }

      this->resam->copy(as_1, s_1);
      if (r_1) {
        subrange(lws2_1, block * blockSize, blockSize).clear();
      }
      this->predict(rng, T, s_1);
      if (block == 0) {
        /* overwrite first particle with conditioned particle */
        row(s_1.getDyn(), 0) = column(X, n);
      }
      this->correct(s_1, subrange(lws2_1, block * blockSize, blockSize));

      // now for filter 2
      this->top();
      if (block == 0) {
        r_2 = resample(rng, lws_2, as_2, false, slws_2, ps_2, Ws_2);
      } else {
        r_2 = resample(rng, lws_2, as_2, true, slws_2, ps_2, Ws_2);
      }
      this->resam->copy(as_2, s_2);
      if (r_2) {
        subrange(lws2_2, block * blockSize, blockSize).clear();
      }
      this->predict(rng, T, s_2);
      this->correct(s_2, subrange(lws2_2, block * blockSize, blockSize));

      if (block == 0) {
        // possibly this should be changed
        maxlw = bi::max(this->getMaxLogWeight(s_1),
            this->getMaxLogWeight(s_2));
      }

      if (stopper->stop(lws2_1, lws2_2, totalObs, maxlw, blockSize)) {
//      if (stoppingRule(lws2_1, lws2_2, T, maxlw)) {
        finished = true;
        this->pop();
        if (block == 0) {

          lws_1.resize(blockSize);
          lws_1 = subrange(lws2_1_base, 0, blockSize);
          as_1_in.resize(blockSize);
          as_1_in = as_1;

          lws_2.resize(blockSize);
          lws_2 = subrange(lws2_2_base, 0, blockSize);
          as_2_in.resize(blockSize);
          as_2_in = as_2;

          s_1.setRange(0, blockSize);
          s_2.setRange(0, blockSize);
        } else {
          int length = block * blockSize;
          s_1.setRange(0, length);
          s_2.setRange(0, length);

          lws_1.resize(length);
          lws_1 = subrange(lws2_1_base, 0, length);
          as_1_in.resize(length);
          as_1_in = subrange(as_1_base, 0, length);

          lws_2.resize(length);
          lws_2 = subrange(lws2_2_base, 0, length);
          as_2_in.resize(length);
          as_2_in = subrange(as_2_base, 0, length);

          currentRange = length;
        }
      } else {
        block++;

        currentRange = (block + 1) * blockSize;

        s_1.setRange(block * blockSize, xvars_1.size1());
        s_1.getDyn() = xvars_1;

        s_2.setRange(block * blockSize, xvars_2.size1());
        s_2.getDyn() = xvars_2;
      }
    }
  }
  return r_2;
}

template<class B, class S, class R, class S2, class IO1>
template<class V1, class V2, class V3, class V4>
bool bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::resample(Random& rng, V1 lws,
    V2 as, bool sorted, V3 slws, V4 ps, V3 Ws) {
  /* pre-condition */
//  BI_ASSERT(s.size() == lws.size());
  int blockSize = as.size();

  bool r =
      this->resam != NULL
          && (this->essRel >= 1.0
              || ess_reduce(lws) <= lws.size() * this->essRel);
//  bool r = true;
  if (r) {
    this->resam->ancestors(rng, lws, as, blockSize, sorted, slws, ps, Ws);
  } else {
    seq_elements(as, 0);
  }
  this->normalise(lws);
  return r;
}

template<class B, class S, class R, class S2, class IO1>
template<class V1, class V2, class V3, class V4>
bool bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::resample(Random& rng, int a,
    V1 lws, V2 as, bool sorted, V3 slws, V4 ps, V3 Ws) {
  /* pre-condition */
//  BI_ASSERT(s.size() == lws.size());
  int blockSize = as.size();

  bool r =
      this->resam != NULL
          && (this->essRel >= 1.0
              || ess_reduce(lws) <= lws.size() * this->essRel);
//  bool r = true;
  if (r) {
    this->resam->ancestors(rng, lws, as, blockSize, a, a, sorted, slws, ps,
        Ws);
//    as[0] = a;
  } else {
    seq_elements(as, 0);
  }
  this->normalise(lws);
  return r;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class M1>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter_impl(Random& rng,
    const real T, const int K, State<B,L>& s, const bool conditional, M1* X) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());
  BI_ASSERT(this->essRel >= 0.0 && this->essRel <= 1.0);

  const int totalObs = this->getSim()->getObs()->countTimes(
      this->getSim()->getTime(), T);
  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);
  lws.clear();

  this->output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : T * k / K;

    /* advance */
    do {
      r = step(rng, tk, totalObs, s, lws, as, n, conditional, X);
      BI_ASSERT(lws.size() == s.size());
      ll += logsumexp_reduce(lws) - bi::log(static_cast<real>(lws.size()));
      this->output(n++, s, r, lws, as);
    } while (this->getSim()->getTime() < tk);

    ++k;
  } while (k <= K);
  this->term();

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class M1>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::filter_impl(Random& rng,
    const real T, const int K, State<B,L>& s_1, M1 X, State<B,L>& s_2,
    real& ll1, real& ll2) {
  /* pre-conditions */
  BI_ASSERT(T >= this->getSim()->getTime());
  BI_ASSERT(this->essRel >= 0.0 && this->essRel <= 1.0);

  const int totalObs = this->getSim()->getObs()->countTimes(
      this->getSim()->getTime(), T);
  const int P = s_1.size();
  int k = 0, n = 0, r_1 = 0, r_2 = 0;

  typename loc_temp_vector<L,real>::type lws_1(P);
  typename loc_temp_vector<L,int>::type as_1(P);

  typename loc_temp_vector<L,real>::type lws_2(P);
  typename loc_temp_vector<L,int>::type as_2(P);

  real tk, lr = 0.0;
  ll1 = 0.0;
  ll2 = 0.0;

  lws_1.clear();
  lws_2.clear();

  this->output0(s_2);
  do {
    /* time of next output */
    tk = (k == K) ? T : T * k / K;

    /* advance */
    do {
      r_2 = step(rng, T, totalObs, s_1, X, s_2, lws_1, as_1, lws_2, as_2, n);

      real ll1inc = logsumexp_reduce(lws_1);
      real ll2inc = logsumexp_reduce(lws_2);
      ll1 += ll1inc - bi::log(static_cast<real>(lws_1.size()));
      ll2 += ll2inc - bi::log(static_cast<real>(lws_2.size()));

      lr += ll2inc - ll1inc;

      this->output(n++, s_2, r_2, lws_2, as_2);
    } while (this->getSim()->getTime() < tk);

    ++k;
  } while (k <= K);
  this->term();

  return lr;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class V2, class M1>
int bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::step_impl(Random& rng,
    const real T, const int totalObs, State<B,L>& s, V1& lws, V2& as_in,
    int n, const bool conditional, M1* X) {

  int P = s.size();

  int maxParticles = stopper->getMaxParticles();
  typename loc_temp_vector<L,int>::type as_base(bi::max(maxParticles, P));
  typename loc_temp_vector<L,real>::type lws2_base(bi::max(maxParticles, P));
  int currentRange = P;

  int block = 0;
  real maxlw = -1.0 / 0.0;
  int r;

  // inside loop

  int lastSize = s.size();
  typename loc_temp_matrix<L,real>::type xvars(lastSize, s.getDyn().size2());
  xvars = s.getDyn();

  typename sim_temp_vector<V1>::type slws(P);
  typename sim_temp_vector<V1>::type Ws(P);
  typename sim_temp_vector<V2>::type ps(P);

  bool finished = false;
  if (n == 0) {
    this->predict(rng, T, s);
    /* overwrite first particle with conditioned particle */
    if (conditional) {
      row(s.getDyn(), 0) = column(*X, n);
    }
    this->correct(s, lws);
  } else {
    currentRange = blockSize;
    subrange(as_base, 0, blockSize).clear();
    subrange(lws2_base, 0, blockSize).clear();
    this->mark();

    while (!finished) {
      BOOST_AUTO(as, subrange(as_base, block * blockSize, blockSize));
      BOOST_AUTO(lws2, subrange(lws2_base, 0, currentRange));

      this->top();
      if (conditional) {
        if (block == 0) {
          r = resample(rng, 0, lws, as, false, slws, ps, Ws);
        } else {
          r = resample(rng, lws, as, true, slws, ps, Ws);
        }
      } else {
        if (block == 0) {
          r = resample(rng, lws, as, false, slws, ps, Ws);
        } else {
          r = resample(rng, lws, as, true, slws, ps, Ws);
        }
      }

      BI_ASSERT(min_reduce(as) >= 0);
      BI_ASSERT(max_reduce(as) < lastSize);

      this->resam->copy(as, s);
      if (r) {
        subrange(lws2, block * blockSize, blockSize).clear();
      }

      this->predict(rng, T, s);
      if (conditional) {
        if (block == 0) {
          /* overwrite first particle with conditioned particle */
          row(s.getDyn(), 0) = column(*X, n);
        }
      }
      if (block == 0) {
        maxlw = this->getMaxLogWeight(s);
      }

      this->correct(s, subrange(lws2, block * blockSize, blockSize));

      if (stopper->stop(lws2, totalObs, maxlw, blockSize)) {
        finished = true;
        this->pop();
        if (block == 0) {
          lws.resize(blockSize);
          lws = subrange(lws2_base, 0, blockSize);
          as_in.resize(blockSize);
          as_in = as;

          // as is already of length blockSize
          s.setRange(0, blockSize);
          s.resizeMax(blockSize, true);
        } else {
          int length = block * blockSize;
          s.setRange(0, length);
          lws.resize(length);
          lws = subrange(lws2_base, 0, length);
          as_in.resize(length);
          as_in = subrange(as_base, 0, length);

          currentRange = length;
          s.resizeMax(length, true);
        }
      } else {
        block++;
        currentRange = (block + 1) * blockSize;
        s.setRange(block * blockSize, xvars.size1());
        s.getDyn() = xvars;
      }
    }
  }
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L>
real bi::AdaptiveNParticleFilter<B,S,R,S2,IO1>::getMaxLogWeight(
    State<B,L>& s) {
  typename loc_temp_vector<L,real>::type maxlw(1);
  maxlw.clear();

  const int P = s.size();

  s.setRange(0, 1);
  if (this->getSim()->getObs()->getTime() == this->getSim()->getTime()) {
    this->m.observationMaxLogDensities(s, this->getSim()->getObs()->getMask(),
        maxlw);
  }
  s.setRange(0, P);

  return *maxlw.begin();
}

#endif
