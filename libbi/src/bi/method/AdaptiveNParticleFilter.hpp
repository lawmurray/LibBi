/*
 * AdaptiveNParticleFilter.hpp
 *
 *  Created on: 28/05/2012
 *      Author: lee27x
 */

#ifndef ADAPTIVENPARTICLEFILTER_HPP_
#define ADAPTIVENPARTICLEFILTER_HPP_

#include "ParticleFilter.hpp"

namespace bi {
/**
 * Adaptive N Particle filter.
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

template<class B, class R, class IO1, class IO2, class IO3, Location CL =
    ON_HOST>
class AdaptiveNParticleFilter: public ParticleFilter<B, R, IO1, IO2, IO3, CL> {
public:
  AdaptiveNParticleFilter(B& m, R* resam = NULL, const real essRel = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~AdaptiveNParticleFilter();

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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO4>
  real filter(Random& rng, const real T, State<B, L>& s, IO4* inInit);

protected:
  template<class V1>
  bool stoppingRule(V1 lws);

  template<class V1, class V2>
  bool resample(Random& rng, V1 lws, V2 as);

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
template<Location CL = ON_HOST>
struct AdaptiveNParticleFilterFactory {
  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptiveNParticleFilter object. Caller has ownership.
   *
   * @see AdaptiveNParticleFilter::AdaptiveNParticleFilter()
   */
  template<class B, class R, class IO1, class IO2, class IO3>
  static AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>* create(B& m,
      R* resam = NULL, const real essRel = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>(m, resam,
        essRel, in, obs, out);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
bi::AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>::AdaptiveNParticleFilter(
    B& m, R* resam, const real essRel, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B, R, IO1, IO2, IO3, CL>(m, resam, essRel, in, obs, out) {
//  std::cerr << "Adaptive N constructor!" << std::endl;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
bi::AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>::~AdaptiveNParticleFilter() {
  this->flush();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class IO4>
real bi::AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>::filter(Random& rng,
    const real T, State<B, L>& s, IO4* inInit) {
  real ll;
  for (int i = 0; i < 1; i++) {
//    if (i % 100 == 0) {
      std::cerr << i << std::endl;
//    }
    /* pre-conditions */
    assert(T >= this->state.t);
    assert(this->essRel >= 0.0 && this->essRel <= 1.0);

    const int P = s.size();
    int n = 0, r = 0;

    typename loc_temp_vector<L, real>::type lws(P);
    typename loc_temp_vector<L, int>::type as(P);
    typename loc_temp_vector<L, real>::type lws2(P);
//  std::vector<int> Ps;

//  std::cerr << T << std::endl;

    ll = 0.0;
    init(rng, s, lws, as, inInit);

    lws2.clear();

    int block;
    int blockSize = std::min(512, P);
    while (this->getTime() < T) {
      block = 0;

      typename loc_temp_matrix<L, real>::type xvars(s.size(), s.ND + s.NR);
      xvars = s.getDyn();
      bool finished = false;
      as.resize(blockSize, false);
      while (!finished) {
        if (n == 0) {
          predict(rng, T, s);
          correct(s, lws);
          finished = true;
//        std::cerr << "Used " << s.size() << " particles." << std::endl;
        } else {
          this->mark();
          r = resample(rng, lws, subrange(as, block * blockSize, blockSize));
          this->resam->copy(subrange(as, block * blockSize, blockSize), s);
          if (r) {
            subrange(lws2, block * blockSize, blockSize).clear();
          }
          predict(rng, T, s);
          correct(s, subrange(lws2, block * blockSize, blockSize));

          if (stoppingRule(subrange(lws2, 0, (block + 1) * blockSize))) {
            finished = true;
//          std::cerr << "Used " << (block+1)*blockSize << " particles." << std::endl;
            if (block == 0) {
              lws.resize(lws2.size(), false);
              lws = lws2;
              lws2.clear();

              as.resize(lws2.size(), true);

              s.setRange(0, lws2.size());
            } else {

              s.setRange(0, block * blockSize);

              lws.resize(s.size(), false);
              lws2.resize(s.size(), true);

              lws = lws2;
              lws2.resize(blockSize);
              lws2.clear();

              as.resize(s.size(), true);
            }
          } else {
            block++;
            s.setRange(block * blockSize, xvars.size1());
            s.getDyn() = xvars;
            lws2.resize((block + 1) * blockSize, true);
            as.resize((block + 1) * blockSize, true);
            this->restore();
          }
        }
      }

      std::cout << s.size() << " ";

      double llinc = logsumexp_reduce(lws) - std::log(s.size());
//    std::cerr << "n = " << n << ", t = " << this->getTime() <<", ll increment = " << llinc << std::endl;
      ll += logsumexp_reduce(lws) - std::log(s.size());

      this->output(n, s, r, lws, as);
      ++n;
    }

    synchronize();
    this->term();
    std::cout << ll << std::endl;

    this->reset();
  }

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<class V1>
bool bi::AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>::stoppingRule(
    V1 lws) {
  typedef typename V1::value_type T1;
  T1 lW = logsumexp_reduce(lws);
//  std::cerr << "lW = " << lW << std::endl;
//  std::cerr << "threshold = " << log(1024.0) << std::endl;
  if (lW >= log(1024.0)) {
    return true;
  } else {
    return false;
  }
//  return true;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<class V1, class V2>
bool bi::AdaptiveNParticleFilter<B, R, IO1, IO2, IO3, CL>::resample(Random& rng,
    V1 lws, V2 as) {
  /* pre-condition */
//  assert (s.size() == lws.size());
  int blockSize = as.size();

  bool r = this->resam != NULL
      && (this->essRel >= 1.0 || ess_reduce(lws) <= lws.size() * this->essRel);
//  bool r = true;
  if (r) {
    this->resam->ancestors(rng, lws, as, blockSize);
  } else {
    seq_elements(as, 0);
  }
  normalise(lws);
  return r;
}

#endif /* ADAPTIVENPARTICLEFILTER_HPP_ */
