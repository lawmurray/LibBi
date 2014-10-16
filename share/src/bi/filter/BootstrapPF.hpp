/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_BOOTSTRAPPF_HPP
#define BI_FILTER_BOOTSTRAPPF_HPP

#include "Filter.hpp"
#include "../simulator/Simulator.hpp"
#include "../state/BootstrapPFState.hpp"
#include "../cache/BootstrapPFCache.hpp"

namespace bi {
/**
 * Bootstrap particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam F Forcer type.
 * @tparam O Observer type.
 * @tparam R Resampler type.
 */
template<class B, class F, class O, class R>
class BootstrapPF: public Simulator<B,F,O> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Forcer.
   * @param obs Observer.
   * @param resam Resampler.
   */
  BootstrapPF(B& m, F& in, O& obs, R& resam);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Resample, predict and correct.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      S1& s, IO1& out);

  /**
   * Sample single particle trajectory.
   *
   * @tparam M1 Matrix type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param[out] X Trajectory.
   * @param out Output buffer.
   *
   * Sample a single particle trajectory from the smooth distribution.
   *
   * On output, @p X is arranged such that rows index variables and columns
   * index times.
   */
  template<class M1, class IO1>
  void samplePath(Random& rng, M1 X, IO1& out);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Update particle weights using observations at the current time.
   *
   * @tparam S1 State type.
   *
   * @param rng Random number generator.
   * @param now Current step in time schedule.
   * @param s State.
   */
  template<class S1>
  void correct(Random& rng, const ScheduleElement now, S1& s);

  /**
   * Resample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void resample(Random& rng, const ScheduleElement now, S1& s);

  /**
   * Finalise.
   */
  template<class S1>
  void term(S1& s);
  //@}

protected:
  /**
   * Compute the maximum log-weight of a particle at the current time.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   *
   * @return Maximum log-weight.
   */
  template<class S1>
  double getMaxLogWeight(const ScheduleElement now, S1& s);

  /**
   * Resampler.
   */
  R& resam;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"
#include "../traits/resampler_traits.hpp"

template<class B, class F, class O, class R>
bi::BootstrapPF<B,F,O,R>::BootstrapPF(B& m, F& in, O& obs, R& resam) :
    Simulator<B,F,O>(m, in, obs), resam(resam) {
  //
}

template<class B, class F, class O, class R>
template<class M1, class IO1>
void bi::BootstrapPF<B,F,O,R>::samplePath(Random& rng, M1 X, IO1& out) {
  if (out.size() > 0) {
    int p = rng.multinomial(out.getLogWeights());
    out.readPath(p, X);
  }
}

template<class B, class F, class O, class R>
template<class S1, class IO1>
void bi::BootstrapPF<B,F,O,R>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) {
  do {
    this->resample(rng, *iter, s);
    ++iter;
    this->predict(rng, *iter, s);
    this->correct(rng, *iter, s);
    this->output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BootstrapPF<B,F,O,R>::correct(Random& rng, const ScheduleElement now,
    S1& s) {
  if (now.isObserved()) {
    this->m.observationLogDensities(s, this->obs.getMask(now.indexObs()),
        s.logWeights());
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BootstrapPF<B,F,O,R>::resample(Random& rng,
    const ScheduleElement now, S1& s) {
  double lW;
  if (resam.isTriggered(now, s, &lW)) {
    if (resampler_needs_max<R>::value && now.isObserved()) {
      resam.setMaxLogWeight(getMaxLogWeight(now, s));
    }
    typename precompute_type<R,S1::location>::type pre;
    resam.precompute(s.logWeights(), pre);
    if (now.hasOutput()) {
      resam.ancestorsPermute(rng, s.logWeights(), s.ancestors(), pre);
      resam.copy(s.ancestors(), s.getDyn());
    } else {
      typename S1::temp_int_vector_type as1(s.ancestors().size());
      resam.ancestorsPermute(rng, s.logWeights(), as1, pre);
      resam.copy(as1, s.getDyn());
      bi::gather(as1, s.ancestors(), s.ancestors());
    }
    s.logWeights().clear();
    s.logLikelihood += lW;
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BootstrapPF<B,F,O,R>::term(S1& s) {
  s.logLikelihood += logsumexp_reduce(s.logWeights())
      - bi::log(static_cast<double>(s.size()));
  Simulator<B,F,O>::term(s);
}

template<class B, class F, class O, class R>
template<class S1>
double bi::BootstrapPF<B,F,O,R>::getMaxLogWeight(const ScheduleElement now,
    S1& s) {
  return this->m.observationMaxLogDensity(s,
      this->obs.getMask(now.indexObs()));
}

#endif
