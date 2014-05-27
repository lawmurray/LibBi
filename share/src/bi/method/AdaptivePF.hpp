/**
 * @file
 *
 * @author
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_ADAPTIVEPF_HPP
#define BI_METHOD_ADAPTIVEPF_HPP

#include "BootstrapPF.hpp"
#include "../resampler/Resampler.hpp"
#include "../stopper/Stopper.hpp"
#include "../state/BootstrapPFState.hpp"

namespace bi {
/**
 * Adaptive particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R Resampler type.
 * @tparam S2 Stopper type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class R, class S2, class IO1>
class AdaptivePF: public BootstrapPF<B,S,R,IO1> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   * @param stopper Stopping criterion for adapting number of particles.
   * @param out Output.
   */
  AdaptivePF(B& m, S* sim = NULL, R* resam = NULL, S2* stopper = NULL,
      IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::filter(Random&, const ScheduleIterator, const ScheduleIterator, BootstrapPFState<B,L>&, IO2*)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, BootstrapPFState<B,L>& s, IO2* inInit);

  /**
   * @copydoc BootstrapPF::filter(Random&, const ScheduleIterator, const ScheduleIterator, BootstrapPFState<B,L>&, IO2*)
   */
  template<Location L, class V1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, BootstrapPFState<B,L>& s);

  /**
   * @copydoc BootstrapPF::step(Random&, ScheduleIterator&, const ScheduleIterator, BootstrapPFState<B,L>&)
   */
  template<bi::Location L>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      BootstrapPFState<B,L>& s);
  //@}

private:
  /**
   * Stopping criterion.
   */
  S2* stopper;
};

/**
 * Factory for creating AdaptivePF objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see AdaptivePF
 */
struct AdaptivePFFactory {
  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptivePF object. Caller has ownership.
   *
   * @see AdaptivePF::AdaptivePF()
   */
  template<class B, class S, class R, class S2, class IO1>
  static AdaptivePF<B,S,R,S2,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, S2* stopper = NULL, IO1* out = NULL) {
    return new AdaptivePF<B,S,R,S2,IO1>(m, sim, resam, stopper, out);
  }

  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptivePF object. Caller has ownership.
   *
   * @see AdaptivePF::AdaptivePF()
   */
  template<class B, class S, class R, class S2>
  static AdaptivePF<B,S,R,S2,BootstrapPFCache<> >* create(B& m, S* sim = NULL,
      R* resam = NULL, S2* stopper = NULL) {
    return new AdaptivePF<B,S,R,S2,BootstrapPFCache<> >(m, sim, resam,
        stopper);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class S2, class IO1>
bi::AdaptivePF<B,S,R,S2,IO1>::AdaptivePF(B& m, S* sim, R* resam, S2* stopper,
    IO1* out) :
    BootstrapPF<B,S,R,IO1>(m, sim, resam, out), stopper(stopper) {
  //
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class IO2>
real bi::AdaptivePF<B,S,R,S2,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    BootstrapPFState<B,L>& s, IO2* inInit) {
  const int P = s.size();
  real ll;

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, inInit);
  this->output0(s);
  ll = this->correct(*iter, s);
  this->output(*iter, s);
  if (this->out != NULL) {
    this->out->push(P);
  }
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1>
real bi::AdaptivePF<B,S,R,S2,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    BootstrapPFState<B,L>& s) {
  const int P = s.size();
  real ll;

  ScheduleIterator iter = first;
  this->init(rng, theta, *iter, s);
  this->output0(s);
  ll = this->correct(*iter, s);
  this->output(*iter, s);
  if (this->out != NULL) {
    this->out->push(P);
  }
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L>
real bi::AdaptivePF<B,S,R,S2,IO1>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, BootstrapPFState<B,L>& s) {
  typedef typename loc_temp_vector<L,real>::type vector_type;
  typedef typename loc_temp_matrix<L,real>::type matrix_type;

  const int P = s.size();
  const int N = s.getDyn().size2();
  const int maxP = stopper->getMaxParticles();
  const int blockP = stopper->getBlockSize();

  matrix_type X(P, N);
  vector_type lws(P);

  X = s.getDyn();
  lws = s.logWeights();

  typename precompute_type<R,L>::type pre;
  this->resam->precompute(lws, pre);

  bool finished = false;
  int block = 0;
  real maxlw, ll = 0.0;
  BOOST_AUTO(iter1, iter);

  /* propagate block by block */
  this->stopper->reset();
  do {
    if (s.sizeMax() < (block + 1) * blockP) {
      s.resizeMax((block + 1) * blockP);
    }
    s.setRange(block * blockP, blockP);
    iter1 = iter;
    this->resam->ancestors(rng, lws, s.ancestors(), pre);
    this->resam->copy(s.ancestors(), X, s.getDyn());
    do {
      ++iter1;
      this->predict(rng, *iter1, s);
      this->output(*iter1, s);
    } while (iter1 + 1 != last && !iter1->isObserved());
    this->correct(*iter1, s);

    if (block == 0) {
      maxlw = this->getMaxLogWeight(*iter1, s);
    }
    ++block;
    finished = stopper->stop(s.logWeights(), maxlw);
  } while (!finished);

  int length = bi::max(block - 1, 1) * blockP;  // drop last block
  if (this->out != NULL) {
    this->out->push(length);
  }
  s.setRange(0, length);
  //s.trim(); // optional, saves memory but means reallocation
  iter = iter1;  // caller expects iter to be advanced at end of step()
  ll = logsumexp_reduce(s.logWeights()) - bi::log(static_cast<real>(length));

  return ll;
}

#endif
