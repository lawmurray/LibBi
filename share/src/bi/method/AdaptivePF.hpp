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
 */
template<class B, class S, class R, class S2>
class AdaptivePF: public BootstrapPF<B,S,R> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   * @param stopper Stopping criterion for adapting number of particles.
   */
  AdaptivePF(B& m, S& sim, R& resam, S2& stopper);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::step()
   */
  template<bi::Location L, class IO1>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      BootstrapPFState<B,L>& s, IO1* out);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * @copydoc BootstrapPF::output()
   */
  template<Location L, class IO1>
  void output(const ScheduleElement now, const BootstrapPFState<B,L>& s,
      IO1* out);
  //@}

private:
  /**
   * Stopping criterion.
   */
  S2& stopper;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class S2>
bi::AdaptivePF<B,S,R,S2>::AdaptivePF(B& m, S& sim, R& resam, S2& stopper) :
    BootstrapPF<B,S,R>(m, sim, resam), stopper(stopper) {
  //
}

template<class B, class S, class R, class S2>
template<bi::Location L, class IO1>
real bi::AdaptivePF<B,S,R,S2>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, BootstrapPFState<B,L>& s, IO1* out) {
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
  this->resam.precompute(lws, pre);

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
    this->resam.ancestors(rng, lws, s.ancestors(), pre);
    this->resam.copy(s.ancestors(), X, s.getDyn());
    do {
      ++iter1;
      this->predict(rng, *iter1, s);
      output(*iter1, s);
    } while (iter1 + 1 != last && !iter1->isObserved());
    this->correct(rng, *iter1, s);

    if (block == 0) {
      maxlw = this->getMaxLogWeight(*iter1, s);
    }
    ++block;
    finished = stopper->stop(s.logWeights(), maxlw);
  } while (!finished);

  int length = bi::max(block - 1, 1) * blockP;  // drop last block
  if (out != NULL) {
    out->push(length);
  }
  s.setRange(0, length);
  //s.trim(); // optional, saves memory but means reallocation
  iter = iter1;  // caller expects iter to be advanced at end of step()
  ll = logsumexp_reduce(s.logWeights()) - bi::log(static_cast<real>(length));

  return ll;
}

template<class B, class S, class R, class S2>
template<bi::Location L, class IO1>
void bi::AdaptivePF<B,S,R,S2>::output(const ScheduleElement now,
    const BootstrapPFState<B,L>& s, IO1* out) {
  BootstrapPF<B,S,R>::output(now, s, out);
  if (out != NULL && now.indexOutput() == 0) {
    /* need to call push()---see AdaptivePFCache---other pushes handled in
     * step() */
    out->push(s.size());
  }
}

#endif
