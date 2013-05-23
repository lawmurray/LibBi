/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_PARTICLEMARGINALMETROPOLISHASTINGS_HPP
#define BI_METHOD_PARTICLEMARGINALMETROPOLISHASTINGS_HPP

#include "../state/Schedule.hpp"
#include "../state/ThetaState.hpp"
#include "../cache/ParticleMCMCCache.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/view.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Particle Marginal Metropolis-Hastings (PMMH) sampler.
 *
 * See @ref Andrieu2010 "Andrieu, Doucet \& Holenstein (2010)".
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam F Filter type.
 * @tparam IO1 Output type.
 */
template<class B, class F, class IO1 = ParticleMCMCCache<> >
class ParticleMarginalMetropolisHastings {
public:
  /**
   * Constructor.
   *
   * @tparam IO2 Input type.
   *
   * @param m Model.
   * @param filter Filter.
   * @param out Output.
   */
  ParticleMarginalMetropolisHastings(B& m, F* filter = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get filter.
   *
   * @return Filter.
   */
  F* getFilter();

  /**
   * Set filter.
   *
   * @param filter Filter.
   */
  void setFilter(F* filter);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output buffer.
   */
  void setOutput(IO1* out);

  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param inInit Initialisation file.
   * @param C Number of samples to draw.
   */
  template<Location L, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, IO2* inInit = NULL,
      const int C = 1, const FilterMode = UNCONDITIONED);

  template<Location L, class IO2>
  void sampleTogether(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, IO2* inInit,
      const int C);
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
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, IO2* inInit = NULL);

  /**
   * Take one step.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param type Type of filtering to perform.
   *
   * @return True if the step is accepted, false otherwise.
   */
  template<Location L>
  bool step(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, const FilterMode type =
          UNCONDITIONED);

  template<Location L, class Q1>
  bool step(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, Q1& q,
      const bool localMove = false, const FilterMode type = UNCONDITIONED);

  template<Location L>
  bool stepTogether(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s);

  template<Location L>
  bool computeAcceptReject(Random& rng, ThetaState<B,L>& s);

  /**
   * Propose using proposal defined in model.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<Location L>
  void propose(Random& rng, ThetaState<B,L>& s);

  /**
   * @tparam L Location.
   * @tparam #concept::Pdf type.
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param q Proposal distribution.
   * @param localMove Should a local move be used? This means that the draw
   * from @p q is added to the local state.
   */
  template<Location L, class Q1>
  void propose(Random& rng, ThetaState<B,L>& s, Q1& q, const bool localMove =
      false);

  /**
   * Update state with log-prior density.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void logPrior(ThetaState<B,L>& s);

  /**
   * Update state with log-likelihood.
   *
   * @tparam L
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param type Type of filtering to perform.
   */
  template<Location L>
  void logLikelihood(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s,
      const FilterMode filtermode = UNCONDITIONED);

  template<Location L>
  real logLikelihood(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaState<B,L>& s, ThetaState<B,L>& s_2);

  /**
   * Output.
   *
   * @tparam L Location.
   *
   * @param c Index in output file.
   * @param s State.
   */
  template<Location L>
  void output(const int c, ThetaState<B,L>& s);

  /**
   * Report progress on stderr.
   *
   * @tparam L Location.
   *
   * @param c Number of steps taken.
   * @param s State.
   */
  template<Location L>
  void report(const int c, ThetaState<B,L>& s);

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
  template<Location L>
  void accept(Random& rng, ThetaState<B,L>& s);

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
  F* filter;

  /**
   * Output buffer.
   */
  IO1* out;

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

/**
 * Factory for creating ParticleMarginalMetropolisHastings objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ParticleMarginalMetropolisHastings
 */
struct ParticleMarginalMetropolisHastingsFactory {
  /**
   * Create particle MCMC sampler.
   *
   * @return ParticleMarginalMetropolisHastings object. Caller has ownership.
   *
   * @see ParticleMarginalMetropolisHastings::ParticleMarginalMetropolisHastings()
   */
  template<class B, class F, class IO1>
  static ParticleMarginalMetropolisHastings<B,F,IO1>* create(B& m, F* filter =
      NULL, IO1* out = NULL) {
    return new ParticleMarginalMetropolisHastings<B,F,IO1>(m, filter, out);
  }

  /**
   * Create particle MCMC sampler.
   *
   * @return ParticleMarginalMetropolisHastings object. Caller has ownership.
   *
   * @see ParticleMarginalMetropolisHastings::ParticleMarginalMetropolisHastings()
   */
  template<class B, class F>
  static ParticleMarginalMetropolisHastings<B,F>* create(B& m, F* filter =
      NULL) {
    return new ParticleMarginalMetropolisHastings<B,F>(m, filter);
  }
};
}

#include "../math/misc.hpp"

template<class B, class F, class IO1>
bi::ParticleMarginalMetropolisHastings<B,F,IO1>::ParticleMarginalMetropolisHastings(
    B& m, F* filter, IO1* out) :
    m(m), filter(filter), out(out), lastAccepted(false), accepted(0), total(0) {
  //
}

template<class B, class F, class IO1>
F* bi::ParticleMarginalMetropolisHastings<B,F,IO1>::getFilter() {
  return filter;
}

template<class B, class F, class IO1>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::setFilter(F* filter) {
  this->filter = filter;
}

template<class B, class F, class IO1>
IO1* bi::ParticleMarginalMetropolisHastings<B,F,IO1>::getOutput() {
  return out;
}

template<class B, class F, class IO1>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class F, class IO1>
template<bi::Location L, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, IO2* inInit, const int C,
    const FilterMode filterMode) {
  /* pre-condition */
  BI_ASSERT(C >= 0);

  const int P = s.size();

  int c;
  init(rng, first, last, s, inInit);
  for (c = 0; c < C; ++c) {
    step(rng, first, last, s, filterMode);
    report(c, s);
    output(c, s);
    s.setRange(0, P);
  }
  term();
}

template<class B, class F, class IO1>
template<bi::Location L, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::sampleTogether(
    Random& rng, const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, IO2* inInit, const int C) {
  /* pre-condition */
  BI_ASSERT(C >= 0);

  const int P = s.size();

  int c;
  init(rng, first, last, s, inInit);

  step(rng, first, last, s, UNCONDITIONED);
  report(0);
  output(0, s);
  s.setRange(0, P);

  for (c = 1; c < C; ++c) {
    stepTogether(rng, first, last, s);
    report(c, s);
    output(c, s);
    s.setRange(0, P);
  }
  term();
}

template<class B, class F, class IO1>
template<bi::Location L, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::init(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, IO2* inInit) {
  /* log-likelihood */
  s.getLogLikelihood1() = filter->filter(rng, first, last, s, inInit);
  s.getParameters1() = vec(s.get(P_VAR));

  /* prior log-density */
  row(s.get(PY_VAR), 0) = s.getParameters1();
  s.getLogPrior1() = m.parameterLogDensity(s);

  /* trajectory */
  filter->sampleTrajectory(rng, s.getTrajectory());

  if (out != NULL) {
    out->clear();
  }
}

template<class B, class F, class IO1>
template<bi::Location L>
bool bi::ParticleMarginalMetropolisHastings<B,F,IO1>::step(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, const FilterMode filtermode) {
  bool result = false;
  try {
    propose(rng, s);
    logPrior(s);
    logLikelihood(rng, first, last, s, filtermode);
    result = computeAcceptReject(rng, s);
  } catch (CholeskyException e) {
    result = false;
  } catch (ParticleFilterDegeneratedException e) {
    result = false;
  } catch (ConditionalParticleFilterException e) {
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

template<class B, class F, class IO1>
template<bi::Location L, class Q1>
bool bi::ParticleMarginalMetropolisHastings<B,F,IO1>::step(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, Q1& q, const bool localMove,
    const FilterMode filtermode) {
  bool result = false;
  try {
    propose(rng, s, q, localMove);
    logPrior(s);
    logLikelihood(rng, first, last, s, filtermode);
    result = computeAcceptReject(rng, s);
  } catch (CholeskyException e) {
    result = false;
  } catch (ParticleFilterDegeneratedException e) {
    result = false;
  } catch (ConditionalParticleFilterException e) {
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

template<class B, class F, class IO1>
template<bi::Location L>
bool bi::ParticleMarginalMetropolisHastings<B,F,IO1>::stepTogether(
    Random& rng, const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s) {
  bool result = false;
  ThetaState<B,L> s_2(m, s.size());
  s_2 = s;

  try {
    propose(rng, s);
    logPrior(s);
    real loglr = logLikelihood(rng, first, last, s, s_2);

    if (!bi::is_finite(s.getLogLikelihood2())) {
      result = false;
    } else if (!bi::is_finite(s.getLogLikelihood1())) {
      result = true;
    } else {
      real logpr = s.getLogPrior2() - s.getLogPrior1();
      real logqr = s.getLogProposal1() - s.getLogProposal2();
      if (!is_finite(s.getLogProposal1())
          && !is_finite(s.getLogProposal2())) {
        logqr = 0.0;
      }
      real logratio = loglr + logpr + logqr;
      real u = rng.uniform<real>();

      result = bi::log(u) < logratio;
    }
  } catch (CholeskyException e) {
    result = false;
  } catch (ParticleFilterDegeneratedException e) {
    result = false;
  } catch (ConditionalParticleFilterException e) {
    result = false;
  }

  /* accept or reject */
  if (result) {
    accept(rng, s_2);
    s = s_2;
  } else {
    reject();
  }

  return result;
}

template<class B, class F, class IO1>
template<bi::Location L>
bool bi::ParticleMarginalMetropolisHastings<B,F,IO1>::computeAcceptReject(
    Random& rng, ThetaState<B,L>& s) {
  bool result;

  if (!bi::is_finite(s.getLogLikelihood2())) {
    result = false;
  } else if (!bi::is_finite(s.getLogLikelihood1())) {
    result = true;
  } else {
    real loglr = s.getLogLikelihood2() - s.getLogLikelihood1();
    real logpr = s.getLogPrior2() - s.getLogPrior1();
    real logqr = s.getLogProposal1() - s.getLogProposal2();

    if (!bi::is_finite(s.getLogProposal1())
        && !bi::is_finite(s.getLogProposal2())) {
      logqr = 0.0;
    }
    real logratio = loglr + logpr + logqr;
    real u = rng.uniform<real>();

    result = bi::log(u) < logratio;
  }

  return result;
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::propose(Random& rng,
    ThetaState<B,L>& s) {
  /* proposal */
  row(s.get(P_VAR), 0) = s.getParameters1();
  m.proposalParameterSample(rng, s);
  s.getParameters2() = row(s.get(P_VAR), 0);

  /* reverse proposal log-density */
  row(s.get(P_VAR), 0) = s.getParameters2();
  row(s.get(PY_VAR), 0) = s.getParameters1();
  s.getLogProposal1() = m.proposalParameterLogDensity(s);

  /* proposal log-density */
  row(s.get(P_VAR), 0) = s.getParameters1();
  row(s.get(PY_VAR), 0) = s.getParameters2();
  s.getLogProposal2() = m.proposalParameterLogDensity(s);
}

template<class B, class F, class IO1>
template<bi::Location L, class Q1>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::propose(Random& rng,
    ThetaState<B,L>& s, Q1& q, const bool localMove) {
  if (localMove) {
    q.sample(rng, s.getParameters2());
    s.getLogProposal2() = q.logDensity(s.getParameters2());
    axpy(1.0, s.getParameters1(), s.getParameters2());

    axpy(-1.0, s.getParameters2(), s.getParameters1());
    s.getLogProposal1() = q.logDensity(s.getParameters1());
    axpy(1.0, s.getParameters2(), s.getParameters1());
  } else {
    q.sample(rng, s.getParameters2());
    s.getLogProposal1() = q.logDensity(s.getParameters1());
    s.getLogProposal2() = q.logDensity(s.getParameters2());
  }
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::logPrior(
    ThetaState<B,L>& s) {
  /* prior log-density */
  row(s.get(PY_VAR), 0) = s.getParameters2();
  s.getLogPrior2() = m.parameterLogDensity(s);
  s.getParameters2() = row(s.get(P_VAR), 0);
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::logLikelihood(
    Random& rng, const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, const FilterMode filtermode) {
  /* log-likelihood */
  if (!bi::is_finite(s.getLogPrior2())) {
    s.getLogLikelihood2() = -1.0 / 0.0;
  } else {
    /* likelihood */
    s.getLogLikelihood2() = filter->filter(rng, first, last,
        s.getParameters2(), s);

    /* check for s.getLogPrior1() is 0 to avoid computation of ll at step 1
     this is to prevent numerical instability associated with a
     (virtual) trajectory with no weight. */
    if (filtermode == CONDITIONED && bi::exp(s.getLogPrior1()) > 0) {
      try {
        s.getLogLikelihood1() = filter->filter(rng, first, last,
            s.getParameters1(), s);
      } catch (CholeskyException e) {
        throw ConditionalParticleFilterException();
      } catch (ParticleFilterDegeneratedException e) {
        throw ConditionalParticleFilterException();
      }
    }
  }
}

template<class B, class F, class IO1>
template<bi::Location L>
real bi::ParticleMarginalMetropolisHastings<B,F,IO1>::logLikelihood(
    Random& rng, const ScheduleIterator first, const ScheduleIterator last,
    ThetaState<B,L>& s, ThetaState<B,L>& s_2) {
  real ll1 = 0.0, ll2 = 0.0, loglr;

  loglr = filter->filter(rng, first, last, s.getParameters1(), s,
      s.getTrajectory(), s.getParameters2(), s_2, ll1, ll2);
  s.getLogLikelihood1() = ll1;
  s.getLogLikelihood2() = ll2;

  return loglr;
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::accept(Random& rng,
    ThetaState<B,L>& s) {
  std::swap(s.getParameters1(), s.getParameters2());
  std::swap(s.getLogLikelihood1(), s.getLogLikelihood2());
  std::swap(s.getLogPrior1(), s.getLogPrior2());
  std::swap(s.getLogProposal1(), s.getLogProposal2());
  filter->sampleTrajectory(rng, s.getTrajectory());

  ++accepted;
  ++total;
  lastAccepted = true;
}

template<class B, class F, class IO1>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::reject() {
  ++total;
  lastAccepted = false;
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::output(const int c,
    ThetaState<B,L>& s) {
  if (out != NULL) {
    if (c == 0) {
      out->writeTimes(0, getFilter()->getOutput()->getTimes());
    }
    out->writeLogLikelihood(c, s.getLogLikelihood1());
    out->writeLogPrior(c, s.getLogPrior1());
    out->writeParameter(c, s.getParameters1());
    out->writeTrajectory(c, s.getTrajectory());
    if (out->isFull()) {
      out->flush();
      out->clear();
    }
  }
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::report(const int c,
    ThetaState<B,L>& s) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << s.getLogLikelihood1();
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.getLogPrior1();
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.getLogProposal1();
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << s.getLogLikelihood2();
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.getLogPrior2();
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.getLogProposal2();
  std::cerr << '\t';
  if (this->wasLastAccepted()) {
    std::cerr << "accept";
  }
  std::cerr << "\taccept=" << (double)accepted / total;
  std::cerr << std::endl;
}

template<class B, class F, class IO1>
void bi::ParticleMarginalMetropolisHastings<B,F,IO1>::term() {
  //
}

template<class B, class F, class IO1>
inline bool bi::ParticleMarginalMetropolisHastings<B,F,IO1>::wasLastAccepted() {
  return lastAccepted;
}

template<class B, class F, class IO1>
inline int bi::ParticleMarginalMetropolisHastings<B,F,IO1>::getNumSteps() {
  return total;
}

template<class B, class F, class IO1>
inline int bi::ParticleMarginalMetropolisHastings<B,F,IO1>::getNumAccepted() {
  return accepted;
}

#endif
