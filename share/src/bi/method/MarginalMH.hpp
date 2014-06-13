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
   * Initialise.
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
   * Propose.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] theta2 State.
   */
  template<class S1>
  void propose(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& theta2);

  /**
   * Accept or reject proposal.
   */
  template<class S1>
  bool acceptReject(Random& rng, S1& theta1, S1& theta2);

  /**
   * Extend time schedule.
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

  init(rng, first, last, s.theta1, inInit);
  for (int c = 0; c < C; ++c) {
    propose(rng, first, last, theta1, theta2);
    acceptReject(rng, theta1, theta2);
    report(c, s);
    output(c, s, out);
  }
  term();
}

template<class B, class F>
template<class S1, class IO2>
void bi::MarginalMH<B,F>::init(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO2& inInit) {
  /* log-likelihood */
  s.theta1.logLikelihood = filter.filter(rng, first, last, s.sFilter,
      s.outFilter, inInit);
  s.theta1 = vec(s.sFilter.get(P_VAR));

  /* prior log-density */
  row(s.sFilter.get(PY_VAR), 0) = s.theta1;
  s.theta1.logPrior = m.parameterLogDensity(s.sFilter);

  /* trajectory */
  filter.samplePath(rng, s.path, s.outFilter);
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::propose(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s) {
  /* proposal */
  row(s.sFilter.get(P_VAR), 0) = s.theta1;
  m.proposalParameterSample(rng, s.sFilter);
  s.theta2 = row(s.sFilter.get(P_VAR), 0);

  /* reverse proposal log-density */
  row(s.sFilter.get(P_VAR), 0) = s.theta2;
  row(s.sFilter.get(PY_VAR), 0) = s.theta1;
  s.theta1.logProposal = m.proposalParameterLogDensity(s.sFilter);

  /* proposal log-density */
  row(s.sFilter.get(P_VAR), 0) = s.theta1;
  row(s.sFilter.get(PY_VAR), 0) = s.theta2;
  s.theta2.logProposal = m.proposalParameterLogDensity(s.sFilter);

  /* prior log-density */
  row(s.sFilter.get(PY_VAR), 0) = s.theta2;
  s.theta2.logPrior = m.parameterLogDensity(s.sFilter);
  s.theta2 = row(s.sFilter.get(P_VAR), 0);

  /* log-likelihood */
  s.theta2.logLikelihood = -1.0 / 0.0;
  if (bi::is_finite(s.theta2.logPrior)) {
    try {
      s.theta2.logLikelihood = filter.filter(rng, s.theta2, first, last,
          s.sFilter, s.outFilter);
    } catch (CholeskyException e) {
      //
    } catch (ParticleFilterDegeneratedException e) {
      //
    } catch (ConditionalBootstrapPFException e) {
      //
    }
  }
}

template<class B, class F>
template<class S1>
bool bi::MarginalMH<B,F>::acceptReject(Random& rng, S1& theta1, S1& theta2) {
  bool result;

  if (!bi::is_finite(s.theta2.logLikelihood)) {
    result = false;
  } else if (!bi::is_finite(s.theta1.logLikelihood)) {
    result = true;
  } else {
    real loglr = s.theta2.logLikelihood - s.theta1.logLikelihood;
    real logpr = s.theta2.logPrior - s.theta1.logPrior;
    real logqr = s.theta1.logProposal - s.theta2.logProposal;

    if (!bi::is_finite(s.theta1.logProposal)
        && !bi::is_finite(s.theta2.logProposal)) {
      logqr = 0.0;
    }
    real logratio = loglr + logpr + logqr;
    real u = rng.uniform<real>();

    result = bi::log(u) < logratio;
  }

  if (result) {
    std::swap(s.theta1, s.theta2);
    std::swap(s.theta1.logLikelihood, s.theta2.logLikelihood);
    std::swap(s.theta1.logPrior, s.theta2.logPrior);
    std::swap(s.theta1.logProposal, s.theta2.logProposal);
    filter.samplePath(rng, s.path, s.theta2.out);

    ++accepted;
  }
  lastAccepted = result;
  ++total;

  return result;
}

template<class B, class F>
template<class S1>
void bi::MarginalMH<B,F>::extend(Random& rng, ScheduleIterator iter,
    const ScheduleIterator last, S1& theta2) {
  filter.step(rng, iter, last, theta2, theta2.out);
  filter.samplePath(rng, s.path, s.out);
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
