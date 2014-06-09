/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MARGINALMH_HPP
#define BI_METHOD_MARGINALMH_HPP

#include "../state/Schedule.hpp"
#include "../state/MarginalMHState.hpp"
#include "../cache/MCMCCache.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/view.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Marginal Metropolis-Hastings.
 *
 * @ingroup method_sampler
 *
 * @tparam B Model type
 * @tparam F Filter type.
 *
 * Implements a marginal Metropolis--Hastings sampler, which, when combined
 * with a particle filter, gives the particle marginal Metropolis--Hastings
 * sampler described in @ref Andrieu2010 "Andrieu, Doucet \& Holenstein (2010)".
 *
 * @todo Add proposal adaptation using adapter classes.
 */
template<class B, class F>
class MarginalMH {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param filter Filter.
   */
  MarginalMH(B& m, F& filter);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Sample.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param C Number of samples to draw.
   * @param out Output buffer.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, const int C = 1, IO1& out = NULL,
      IO2& inInit = NULL);
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
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, IO1& out = NULL,
      IO2& inInit = NULL);

  /**
   * Take one step.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   *
   * @return True if the step is accepted, false otherwise.
   */
  template<class S1>
  bool step(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s);

  template<class S1, class Q1>
  bool step(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, Q1& q,
      const bool localMove = false);

  template<class S1>
  bool acceptReject(Random& rng, S1& s);

  /**
   * Propose using proposal defined in model.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<class S1>
  void propose(Random& rng, S1& s);

  /**
   * @tparam S1 State type.
   * @tparam #concept::Pdf type.
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param q Proposal distribution.
   * @param localMove Should a local move be used? This means that the draw
   * from @p q is added to the local state.
   */
  template<class S1, class Q1>
  void propose(Random& rng, S1& s, Q1& q, const bool localMove = false);

  /**
   * Update state with log-prior density.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] s State.
   */
  template<class S1>
  void prior(S1& s);

  /**
   * Update state with log-likelihood.
   *
   * @tparam L
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void likelihood(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s);

  /**
   * Output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param c Index in output file.
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output(const int c, S1& s, IO1& out);

  /**
   * Report progress on stderr.
   *
   * @tparam S1 State type.
   *
   * @param c Number of steps taken.
   * @param s State.
   */
  template<class S1>
  void report(const int c, S1& s);

  /**
   * Terminate.
   */
  void term();
  //@}

  /**
   * @name Diagnostics
   */
  //@{
  /**
   * Was last proposal accepted?
   */
  bool wasLastAccepted();

  /**
   * Get number of steps taken.
   *
   * @return Number of steps taken.
   */
  int getNumSteps();

  /**
   * Get number of accepted proposals.
   *
   * @return Number of accepted proposals.
   */
  int getNumAccepted();

  /**
   * Accept step.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<class S1>
  void accept(Random& rng, S1& s);

  /**
   * Reject step.
   */
  void reject();
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Filter.
   */
  F& filter;

  /**
   * Was last proposal accepted?
   */
  bool lastAccepted;

  /**
   * Number of accepted proposals.
   */
  int accepted;

  /**
   * Total number of proposals.
   */
  int total;
};
}

#include "../math/misc.hpp"

template<class B, class F>
bi::MarginalMH<B,F>::MarginalMH(B& m, F& filter) :
    m(m), filter(filter), lastAccepted(false), accepted(0), total(0) {
  //
}

template<class B, class F>
template<class S1, class IO1, class IO2>
void bi::MarginalMH<B,F>::sample(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, const int C, IO1& out, IO2& inInit) {
  /* pre-condition */
  BI_ASSERT(C >= 0);

  const int P = s.sFilter.size();
  init(rng, first, last, s, out, inInit);
  for (int c = 0; c < C; ++c) {
    step(rng, first, last, s);
    report(c, s);
    output(c, s, out);
    s.sFilter.setRange(0, P);
  }
  term();
}

template<class B, class F>
template<class S1, class IO1, class IO2>
void bi::MarginalMH<B,F>::init(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out, IO2& inInit) {
  /* log-likelihood */
  s.logLikelihood1 = filter.filter(rng, first, last, s.sFilter, s.outFilter,
      inInit);
  s.theta1 = vec(s.sFilter.get(P_VAR));

  /* prior log-density */
  row(s.sFilter.get(PY_VAR), 0) = s.theta1;
  s.logPrior1 = m.parameterLogDensity(s.sFilter);

  /* trajectory */
  filter.samplePath(rng, s.path, s.outFilter);

  out.clear();
}

template<class B, class F>
template<class S1>
bool bi::MarginalMH<B,F>::step(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s) {
  bool result = false;
  try {
    propose(rng, s);
    prior(s);
    likelihood(rng, first, last, s);
    result = acceptReject(rng, s);
  } catch (CholeskyException e) {
    result = false;
  } catch (ParticleFilterDegeneratedException e) {
    result = false;
  } catch (ConditionalBootstrapPFException e) {
    result = false;
  }

  /* accept or reject */
  if (result) {
    accept(rng, s);
  } else {
    reject();
  }
  return result;
}

template<class B, class F>
template<class S1, class Q1>
bool bi::MarginalMH<B,F>::step(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, Q1& q, const bool localMove) {
  bool result = false;
  try {
    propose(rng, s, q, localMove);
    prior(s);
    likelihood(rng, first, last, s);
    result = acceptReject(rng, s);
  } catch (CholeskyException e) {
    result = false;
  } catch (ParticleFilterDegeneratedException e) {
    result = false;
  } catch (ConditionalBootstrapPFException e) {
    result = false;
  }

  /* accept or reject */
  if (result) {
    accept(rng, s);
  } else {
    reject();
  }
  return result;
}

template<class B, class F>
template<class S1>
bool bi::MarginalMH<B,F>::acceptReject(Random& rng, S1& s) {
  bool result;

  if (!bi::is_finite(s.logLikelihood2)) {
    result = false;
  } else if (!bi::is_finite(s.logLikelihood1)) {
    result = true;
  } else {
    real loglr = s.logLikelihood2 - s.logLikelihood1;
    real logpr = s.logPrior2 - s.logPrior1;
    real logqr = s.logProposal1 - s.logProposal2;

    if (!bi::is_finite(s.logProposal1) && !bi::is_finite(s.logProposal2)) {
      logqr = 0.0;
    }
    real logratio = loglr + logpr + logqr;
    real u = rng.uniform<real>();

    result = bi::log(u) < logratio;
  }

  return result;
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::propose(Random& rng, S1& s) {
  /* proposal */
  row(s.sFilter.get(P_VAR), 0) = s.theta1;
  m.proposalParameterSample(rng, s.sFilter);
  s.theta2 = row(s.sFilter.get(P_VAR), 0);

  /* reverse proposal log-density */
  row(s.sFilter.get(P_VAR), 0) = s.theta2;
  row(s.sFilter.get(PY_VAR), 0) = s.theta1;
  s.logProposal1 = m.proposalParameterLogDensity(s.sFilter);

  /* proposal log-density */
  row(s.sFilter.get(P_VAR), 0) = s.theta1;
  row(s.sFilter.get(PY_VAR), 0) = s.theta2;
  s.logProposal2 = m.proposalParameterLogDensity(s.sFilter);
}

template<class B, class F>
template<class S1, class Q1>
void bi::MarginalMH<B,F>::propose(Random& rng, S1& s, Q1& q,
    const bool localMove) {
  if (localMove) {
    q.sample(rng, s.theta2);
    s.logProposal2 = q.logDensity(s.theta2);
    axpy(1.0, s.theta1, s.theta2);

    axpy(-1.0, s.theta2, s.theta1);
    s.logProposal1 = q.logDensity(s.theta1);
    axpy(1.0, s.theta2, s.theta1);
  } else {
    q.sample(rng, s.theta2);
    s.logProposal1 = q.logDensity(s.theta1);
    s.logProposal2 = q.logDensity(s.theta2);
  }
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::prior(S1& s) {
  /* prior log-density */
  row(s.sFilter.get(PY_VAR), 0) = s.theta2;
  s.logPrior2 = m.parameterLogDensity(s.sFilter);
  s.theta2 = row(s.sFilter.get(P_VAR), 0);
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::likelihood(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s) {
  if (!bi::is_finite(s.logPrior2)) {
    s.logLikelihood2 = -1.0 / 0.0;
  } else {
    s.logLikelihood2 = -1.0 / 0.0;  // in case of exception
    s.logLikelihood2 = filter.filter(rng, s.theta2, first, last, s.sFilter,
        s.outFilter);
  }
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::accept(Random& rng, S1& s) {
  std::swap(s.theta1, s.theta2);
  std::swap(s.logLikelihood1, s.logLikelihood2);
  std::swap(s.logPrior1, s.logPrior2);
  std::swap(s.logProposal1, s.logProposal2);
  filter.samplePath(rng, s.path, s.outFilter);

  ++accepted;
  ++total;
  lastAccepted = true;
}

template<class B, class F>
void bi::MarginalMH<B,F>::reject() {
  ++total;
  lastAccepted = false;
}

template<class B, class F>
template<class S1, class IO1>
void bi::MarginalMH<B,F>::output(const int c, S1& s, IO1& out) {
  out.write(c, s);
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::report(const int c, S1& s) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << s.logLikelihood1;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.logPrior1;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.logProposal1;
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << s.logLikelihood2;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.logPrior2;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.logProposal2;
  std::cerr << '\t';
  if (this->wasLastAccepted()) {
    std::cerr << "accept";
  }
  std::cerr << "\taccept=" << (double)accepted / total;
  std::cerr << std::endl;
}

template<class B, class F>
void bi::MarginalMH<B,F>::term() {
  //
}

template<class B, class F>
inline bool bi::MarginalMH<B,F>::wasLastAccepted() {
  return lastAccepted;
}

template<class B, class F>
inline int bi::MarginalMH<B,F>::getNumSteps() {
  return total;
}

template<class B, class F>
inline int bi::MarginalMH<B,F>::getNumAccepted() {
  return accepted;
}

#endif
