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
      const ScheduleIterator last, S1& s, const int C, IO1& out, IO2& inInit);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise starting state.
   *
   * @tparam S1 State type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] theta1 State.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& theta1, IO2& inInit);

  /**
   * Propose new state.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param theta1 Current state.
   * @param[out] theta2 Proposed state.
   */
  template<class S1>
  void propose(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& theta1, S1& theta2);

  /**
   * Accept or reject proposed state.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param theta1 Current state.
   * @param theta2 Proposed state.
   *
   * @return Was the proposal accepted?
   */
  template<class S1>
  bool acceptReject(Random& rng, S1& theta1, S1& theta2);

  /**
   * Extend log-likelihood to next observation.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] theta2 State.
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<class S1>
  double extend(Random& rng, ScheduleIterator& iter,
      const ScheduleIterator last, S1& theta2);

  /**
   * Output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param c Index in output file.
   * @param[out] s State.
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
   * Was the last proposal accepted?
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
  BI_ERROR(C > 0);

  init(rng, first, last, s.theta1, inInit);
  output(0, s, out);
  for (int c = 1; c < C; ++c) {
    propose(rng, first, last, s.theta1, s.theta2);
    acceptReject(rng, s.theta1, s.theta2);
    report(c, s);
    output(c, s, out);
  }
  term();
}

template<class B, class F>
template<class S1, class IO2>
void bi::MarginalMH<B,F>::init(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& theta1, IO2& inInit) {
  /* log-likelihood */
  theta1.logLikelihood = filter.filter(rng, first, last, theta1, theta1.out,
      inInit);

  /* prior log-density */
  theta1.get(PY_VAR) = theta1.get(P_VAR);
  theta1.logPrior = m.parameterLogDensity(theta1);

  /* path */
  filter.samplePath(rng, theta1.path, theta1.out);
  lastAccepted = true;
  accepted = 1;
  total = 1;
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::propose(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& theta1, S1& theta2) {
  /* proposal */
  theta2.get(P_VAR) = theta1.get(P_VAR);
  m.proposalParameterSample(rng, theta2);

  /* reverse proposal log-density */
  theta1.get(PY_VAR) = theta1.get(P_VAR);
  theta1.get(P_VAR) = theta2.get(P_VAR);
  theta1.logProposal = m.proposalParameterLogDensity(theta1);

  /* proposal log-density */
  theta2.get(PY_VAR) = theta2.get(P_VAR);
  theta2.get(P_VAR) = theta1.get(P_VAR);
  theta2.logProposal = m.proposalParameterLogDensity(theta2);

  /* prior log-density */
  theta2.get(PY_VAR) = theta2.get(P_VAR);
  theta2.logPrior = m.parameterLogDensity(theta2);

  /* log-likelihood */
  theta2.logLikelihood = -1.0 / 0.0;
  if (bi::is_finite(theta2.logPrior)) {
    try {
      theta2.logLikelihood = filter.filter(rng, first, last, theta2,
          theta2.out);
    } catch (CholeskyException e) {
      //
    } catch (ParticleFilterDegeneratedException e) {
      //
    }
  }
}

template<class B, class F>
template<class S1>
bool bi::MarginalMH<B,F>::acceptReject(Random& rng, S1& theta1, S1& theta2) {
  if (!bi::is_finite(theta2.logLikelihood)) {
    lastAccepted = false;
  } else if (!bi::is_finite(theta1.logLikelihood)) {
    lastAccepted = true;
  } else {
    double loglr = theta2.logLikelihood - theta1.logLikelihood;
    double logpr = theta2.logPrior - theta1.logPrior;
    double logqr = theta1.logProposal - theta2.logProposal;

    if (!bi::is_finite(theta1.logProposal)
        && !bi::is_finite(theta2.logProposal)) {
      logqr = 0.0;
    }
    double logratio = loglr + logpr + logqr;
    double u = rng.uniform<double>();

    lastAccepted = bi::log(u) < logratio;
  }

  if (lastAccepted) {
    filter.samplePath(rng, theta2.path, theta2.out);
    theta2.swap(theta1);
    ++accepted;
  }
  ++total;

  return lastAccepted;
}

template<class B, class F>
template<class S1>
double bi::MarginalMH<B,F>::extend(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& theta2) {
  double ll = filter.step(rng, iter, last, theta2, theta2.out);
  theta2.logLikelihood += ll;
  if (iter + 1 == last) {
    filter.samplePath(rng, theta2.path, theta2.out);
  }
  return ll;
}

template<class B, class F>
template<class S1, class IO1>
void bi::MarginalMH<B,F>::output(const int c, S1& s, IO1& out) {
  out.write(c, s.theta1);
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::report(const int c, S1& s) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << s.theta1.logLikelihood;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.theta1.logPrior;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.theta1.logProposal;
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << s.theta2.logLikelihood;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.theta2.logPrior;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s.theta2.logProposal;
  std::cerr << '\t';
  if (lastAccepted) {
    std::cerr << "accept";
  }
  std::cerr << "\taccept=" << (double)accepted / total;
  std::cerr << std::endl;
}

template<class B, class F>
void bi::MarginalMH<B,F>::term() {
  //
}

#endif
