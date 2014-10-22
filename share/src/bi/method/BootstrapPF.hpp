/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_BOOTSTRAPPF_HPP
#define BI_METHOD_BOOTSTRAPPF_HPP

#include "Filter.hpp"
#include "Simulator.hpp"
#include "../state/BootstrapPFState.hpp"
#include "../cache/BootstrapPFCache.hpp"

namespace bi {
/**
 * Bootstrap particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R Resampler type.
 */
template<class B, class S, class R>
class BootstrapPF {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   */
  BootstrapPF(B& m, S& sim, R& resam);

  /**
   * @name High-level interface.
   *Got84mp!
   *Got84mp!
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
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<class S1, class IO1>
  double step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
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
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param[out] out Output buffer.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out,
      IO2& inInit);

  /**
   * Initialise.
   *
   * @tparam S1 State type.
   * @tparam V1 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param[out] out Output buffer.
   */
  template<class S1, class IO1>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out);

  /**
   * Predict.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param next Next step in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void predict(Random& rng, const ScheduleElement next, S1& s);

  /**
   * Update particle weights using observations at the current time.
   *
   * @tparam S1 State type.
   *
   * @param rng Random number generator.
   * @param now Current step in time schedule.
   * @param s State.
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<class S1>
  double correct(Random& rng, const ScheduleElement now, S1& s);

  /**
   * Resample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<class S1>
  bool resample(Random& rng, const ScheduleElement now, S1& s);

  /**
   * Output static variables.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output0(const S1& s, IO1& out);

  /**
   * Output dynamic variables.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param now Current step in time schedule.
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output(const ScheduleElement now, const S1& s, IO1& out);

  /**
   * Output marginal log-likelihood estimate.
   *
   * @tparam IO1 Output type.
   *
   * @param ll Estimate of the marginal log-likelihood.
   * @param out Output buffer.
   */
  template<class IO1>
  void outputT(const double ll, IO1& out);

  /**
   * Clean up.
   */
  void term();
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
   * Model.
   */
  B& m;

  /**
   * Simulator.
   */
  S& sim;

  /**
   * Resampler.
   */
  R& resam;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"
#include "../traits/resampler_traits.hpp"

template<class B, class S, class R>
bi::BootstrapPF<B,S,R>::BootstrapPF(B& m, S& sim, R& resam) :
    m(m), sim(sim), resam(resam) {
  //
}

template<class B, class S, class R>
template<class M1, class IO1>
void bi::BootstrapPF<B,S,R>::samplePath(Random& rng, M1 X, IO1& out) {
  if (out.size() > 0) {
    int p = rng.multinomial(out.getLogWeights());
    out.readPath(p, X);
  }
}

template<class B, class S, class R>
template<class S1, class IO1, class IO2>
void bi::BootstrapPF<B,S,R>::init(Random& rng, const ScheduleElement now,
    S1& s, IO1& out, IO2& inInit) {
  sim.init(rng, now, s, inInit);
  s.logWeights().clear();
  seq_elements(s.ancestors(), 0);
  out.clear();
}

template<class B, class S, class R>
template<class S1, class IO1>
void bi::BootstrapPF<B,S,R>::init(Random& rng, const ScheduleElement now,
    S1& s, IO1& out) {
  sim.init(rng, now, s);
  s.logWeights().clear();
  seq_elements(s.ancestors(), 0);
  out.clear();
}

template<class B, class S, class R>
template<class S1, class IO1>
double bi::BootstrapPF<B,S,R>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) {
  double ll = 0.0;
  do {
    resample(rng, *iter, s);
    ++iter;
    predict(rng, *iter, s);
    ll += correct(rng, *iter, s);
    output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());

  return ll;
}

template<class B, class S, class R>
template<class S1>
void bi::BootstrapPF<B,S,R>::predict(Random& rng, const ScheduleElement next,
    S1& s) {
  sim.advance(rng, next, s);
}

template<class B, class S, class R>
template<class S1>
double bi::BootstrapPF<B,S,R>::correct(Random& rng, const ScheduleElement now,
    S1& s) {
  double ll = 0.0;
  if (now.isObserved()) {
    m.observationLogDensities(s, sim.obs.getMask(now.indexObs()),
        s.logWeights());
    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<double>(s.size()));
  }
  return ll;
}

template<class B, class S, class R>
template<class S1>
bool bi::BootstrapPF<B,S,R>::resample(Random& rng, const ScheduleElement now,
    S1& s) {
  bool r = now.isObserved();
  if (r) {
    r = resam.isTriggered(s.logWeights());
    if (r) {
      if (resampler_needs_max<R>::value) {
        resam.setMaxLogWeight(getMaxLogWeight(now, s));
      }
      if (now.hasOutput()) {
        resam.resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
      } else {
        typename S1::temp_int_vector_type as1(s.ancestors().size());
        resam.resample(rng, s.logWeights(), as1, s.getDyn());
        bi::gather(as1, s.ancestors(), s.ancestors());
      }
    } else {
      seq_elements(s.ancestors(), 0);
      Resampler::normalise(s.logWeights());
    }
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
  return r;
}

template<class B, class S, class R>
template<class S1, class IO1>
void bi::BootstrapPF<B,S,R>::output0(const S1& s, IO1& out) {
  out.write0(s);
}

template<class B, class S, class R>
template<class S1, class IO1>
void bi::BootstrapPF<B,S,R>::output(const ScheduleElement now, const S1& s,
    IO1& out) {
  if (now.hasOutput()) {
    out.write(now.indexOutput(), now.getTime(), s);
  }
}

template<class B, class S, class R>
template<class IO1>
void bi::BootstrapPF<B,S,R>::outputT(const double ll, IO1& out) {
  out.writeT(ll);
}

template<class B, class S, class R>
void bi::BootstrapPF<B,S,R>::term() {
  sim.term();
}

template<class B, class S, class R>
template<class S1>
double bi::BootstrapPF<B,S,R>::getMaxLogWeight(const ScheduleElement now,
    S1& s) {
  return this->m.observationMaxLogDensity(s, sim.obs.getMask(now.indexObs()));
}

#endif
