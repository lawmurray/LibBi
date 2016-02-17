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
#include "../misc/exception.hpp"

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
   * Sample single path from filter output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param[out] s State.
   * @param out Output buffer.
   *
   * Sample a single path from the smooth distribution.
   */
  template<class S1, class IO1>
  void samplePath(Random& rng, S1& s, IO1& out);
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
  void resample(Random& rng, const ScheduleElement now, S1& s)
      throw (ParticleFilterDegeneratedException);

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
template<class S1, class IO1>
void bi::BootstrapPF<B,F,O,R>::samplePath(Random& rng, S1& s, IO1& out) {
  if (out.size() > 0) {
    int p = rng.multinomial(out.getLogWeights());
    out.readPath(p, columns(s.path, 0, out.len));
    subrange(s.times, 0, out.len) = out.timeCache.get(0, out.len);
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
    double lW;
    s.ess = resam.reduce(s.logWeights(), &lW);
    s.logIncrements(now.indexObs()) = lW - s.logLikelihood;
    s.logLikelihood = lW;
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BootstrapPF<B,F,O,R>::resample(Random& rng,
    const ScheduleElement now, S1& s)
        throw (ParticleFilterDegeneratedException) {
  resam.resample(rng, now, s);
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BootstrapPF<B,F,O,R>::term(S1& s) {
  s.logLikelihood = logsumexp_reduce(s.logWeights())
      - bi::log(double(s.size()));
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
