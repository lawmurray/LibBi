/**
 * @file
 *
 * @author
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_ADAPTIVEPF_HPP
#define BI_FILTER_ADAPTIVEPF_HPP

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
 * @tparam F Forcer type.
 * @tparam O Observer type.
 * @tparam R Resampler type.
 * @tparam S2 Stopper type.
 */
template<class B, class F, class O, class R, class S2>
class AdaptivePF: public BootstrapPF<B,F,O,R> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Forcer.
   * @param obs Observer.
   * @param resam Resampler.
   * @param stopper Stopping criterion for adapting number of particles.
   * @param initialP Number of particles at first time.
   * @param blockP Number of particles per block.
   */
  AdaptivePF(B& m, F& in, O& obs, R& resam, S2& stopper, const int initialP,
      const int blockP);

  /**
   * @copydoc BootstrapPF::init()
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out,
      IO2& inInit);

  /**
   * @copydoc BootstrapPF::init()
   */
  template<class S1, class IO1>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::step()
   */
  template<class S1, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      S1& s, IO1& out);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * @copydoc BootstrapPF::output()
   */
  template<class S1, class IO1>
  void output(const ScheduleElement now, const S1& s, IO1& out);
  //@}

private:
  /**
   * Stopping criterion.
   */
  S2& stopper;

  /**
   * Initial size.
   */
  int initialP;

  /**
   * Block size.
   */
  int blockP;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class F, class O, class R, class S2>
bi::AdaptivePF<B,F,O,R,S2>::AdaptivePF(B& m, F& in, O& obs, R& resam,
    S2& stopper, const int initialP, const int blockP) :
    BootstrapPF<B,F,O,R>(m, in, obs, resam), stopper(stopper), initialP(
        initialP), blockP(blockP) {
  //
}

template<class B, class F, class O, class R, class S2>
template<class S1, class IO1, class IO2>
void bi::AdaptivePF<B,F,O,R,S2>::init(Random& rng, const ScheduleElement now,
    S1& s, IO1& out, IO2& inInit) {
  if (s.size() < initialP) {
    s.resizeMax(initialP);
  }
  s.setRange(0, initialP);
  BootstrapPF<B,F,O,R>::init(rng, now, s, out, inInit);
}

template<class B, class F, class O, class R, class S2>
template<class S1, class IO1>
void bi::AdaptivePF<B,F,O,R,S2>::init(Random& rng, const ScheduleElement now,
    S1& s, IO1& out) {
  if (s.size() < initialP) {
    s.resizeMax(initialP);
  }
  s.setRange(0, initialP);
  BootstrapPF<B,F,O,R>::init(rng, now, s, out);
}

template<class B, class F, class O, class R, class S2>
template<class S1, class IO1>
void bi::AdaptivePF<B,F,O,R,S2>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) {
  typedef typename loc_temp_vector<S1::location,real>::type vector_type;
  typedef typename loc_temp_matrix<S1::location,real>::type matrix_type;
  typedef typename loc_temp_vector<S1::location,int>::type int_vector_type;

  const int P = s.size();
  const int N = s.getDyn().size2();

  /* state at current time */
  matrix_type X(P, N);
  vector_type lws(P);
  int_vector_type as(P);

  X = s.getDyn();
  lws = s.logWeights();
  as = s.ancestors();

  int block = 0;
  double maxlw, ll = 0.0;
  BOOST_AUTO(iter1, iter);

  /* marginal log-likelihood increment */
  s.logLikelihood += logsumexp_reduce(s.logWeights())
      - bi::log(static_cast<double>(s.size()));

  /* prepare resampler */
  if (iter->isObserved() && resampler_needs_max<R>::value) {
    this->resam.setMaxLogWeight(this->getMaxLogWeight(*iter, s));
  }
  typename precompute_type<R,S1::location>::type pre;
  this->resam.precompute(s.logWeights(), pre);

  /* propagate block by block */
  this->stopper.reset();
  do {
    if (s.sizeMax() < (block + 1) * blockP) {
      s.resizeMax((block + 1) * blockP);
    }
    s.setRange(block * blockP, blockP);
    iter1 = iter;

    do {
      /* resample */
      if (iter1->isObserved() || iter1->indexTime() == 0) {
        if (iter1->hasOutput()) {
          this->resam.ancestors(rng, lws, s.ancestors(), pre);
          this->resam.copy(s.ancestors(), X, s.getDyn());
        } else {
          typename S1::temp_int_vector_type as1(blockP);
          this->resam.ancestors(rng, lws, as1, pre);
          this->resam.copy(as1, X, s.getDyn());
          bi::gather(as1, as, s.ancestors());
        }
        s.logWeights().clear();
      } else if (iter1->hasOutput()) {
        seq_elements(s.ancestors(), block * blockP);
      }

      ++iter1;
      this->predict(rng, *iter1, s);
      this->correct(rng, *iter1, s);
      output(*iter1, s, out);
    } while (iter1 + 1 != last && !iter1->isObserved());

    if (iter1->isObserved()) {  // may not be observed at last time
      if (block == 0) {
        maxlw = this->getMaxLogWeight(*iter1, s);
      }
      stopper.add(s.logWeights(), maxlw);
    }
    ++block;
  } while (iter1->isObserved() && !stopper.stop(maxlw));  // may not be observed at last time

  int length = bi::max(block - 1, 1) * blockP;  // drop last block
  out.push(length);
  s.setRange(0, length);
  //s.trim(); // optional, saves memory but means reallocation
  iter = iter1;  // caller expects iter to be advanced at end of step()
}

template<class B, class F, class O, class R, class S2>
template<class S1, class IO1>
void bi::AdaptivePF<B,F,O,R,S2>::output(const ScheduleElement now,
    const S1& s, IO1& out) {
  if (now.hasOutput()) {
    BootstrapPF<B,F,O,R>::output(now, s, out);
    if (now.indexOutput() == 0) {
      /* need to call push()---see AdaptivePFCache---other pushes handled in
       * step() */
      out.push(s.size());
    }
  }
}

#endif
