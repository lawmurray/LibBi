/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SAMPLER_MARGINALMH_HPP
#define BI_SAMPLER_MARGINALMH_HPP

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
   * @name High-level interface
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
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise starting state.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s1 State.
   * @param[in,out] out Output buffer;
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s1, IO1& out, IO2& inInit);

  /**
   * Propose new state.
   *
   * @tparam S1 State type.
   * @tparam S2 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s1 Current state.
   * @param[out] s2 Proposed state.
   * @param[in,out] out Output buffer.
   */
  template<class S1, class S2, class IO1>
  void propose(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s1, S2& s2, IO1& out);

  /**
   * Accept or reject proposed state.
   *
   * @tparam S1 State type.
   * @tparam S2 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param s1 Current state.
   * @param s2 Proposed state.
   *
   * @return Was proposal accepted?
   */
  template<class S1, class S2, class IO1>
  bool acceptReject(Random& rng, S1& s1, S2& s2, IO1& out);

  /**
   * Output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param c Index in output file.
   * @param s State.
   * @param[in,out] out Output buffer.
   */
  template<class S1, class IO1>
  void output(const int c, const S1& s1, IO1& out);

  /**
   * @copydoc Simulator::outputT()
   */
  template<class S1, class IO1>
  void outputT(const S1& s, IO1& out);

  /**
   * Report progress on stderr.
   *
   * @tparam S1 State type.
   * @tparam S2 State type.
   *
   * @param c Number of steps taken.
   * @param s1 Current state.
   * @param s2 Alternative state.
   */
  template<class S1, class S2>
  void report(const int c, const S1& s1, const S2& s2);

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

#include "../misc/TicToc.hpp"

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

  TicToc clock;
  init(rng, first, last, s.s1, s.out, inInit);
  output(0, s.s1, out);
  for (int c = 1; c < C; ++c) {
    propose(rng, first, last, s.s1, s.s2, s.out);
    acceptReject(rng, s.s1, s.s2, s.out);
    report(c, s.s1, s.s2);
    output(c, s.s1, out);
  }
  s.clock = clock.toc();
  outputT(s, out);
  term();
}

template<class B, class F>
template<class S1, class IO1, class IO2>
void bi::MarginalMH<B,F>::init(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s1, IO1& out, IO2& inInit) {
  filter.init(rng, *first, s1, out, inInit);
  filter.filter(rng, first, last, s1, out);
  filter.samplePath(rng, s1, out);
  lastAccepted = true;
  accepted = 1;
  total = 1;
}

template<class B, class F>
template<class S1, class S2, class IO1>
void bi::MarginalMH<B,F>::propose(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s1, S2& s2, IO1& out) {
  try {
    filter.propose(rng, *first, s1, s2, out);
    if (bi::is_finite(s2.logPrior)) {
      filter.filter(rng, first, last, s2, out);
    } else {
      s2.logLikelihood = -BI_INF;
    }
  } catch (CholeskyException e) {
    s2.logLikelihood = -BI_INF;
  } catch (ParticleFilterDegeneratedException e) {
    s2.logLikelihood = -BI_INF;
  }
}

template<class B, class F>
template<class S1, class S2, class IO1>
bool bi::MarginalMH<B,F>::acceptReject(Random& rng, S1& s1, S2& s2, IO1& out) {
  if (!bi::is_finite(s2.logLikelihood)) {
    lastAccepted = false;
  } else if (!bi::is_finite(s1.logLikelihood)) {
    lastAccepted = true;
  } else {
    double loglr = s2.logLikelihood - s1.logLikelihood;
    double logpr = s2.logPrior - s1.logPrior;
    double logqr = s1.logProposal - s2.logProposal;
    double logratio = loglr + logpr + logqr;
    double u = rng.uniform<double>();

    lastAccepted = bi::log(u) < logratio;
  }

  if (lastAccepted) {
    filter.samplePath(rng, s2, out);
    s2.swap(s1);
    ++accepted;
  }
  ++total;

  return lastAccepted;
}

template<class B, class F>
template<class S1, class IO1>
void bi::MarginalMH<B,F>::output(const int c, const S1& s1, IO1& out) {
  out.write(c, s1);
  if (out.isFull()) {
    out.flush();
    out.clear();
  }
}

template<class B, class F>
template<class S1, class IO1>
void bi::MarginalMH<B,F>::outputT(const S1& s, IO1& out) {
  out.writeClock(s.clock);
}

template<class B, class F>
template<class S1, class S2>
void bi::MarginalMH<B,F>::report(const int c, const S1& s1, const S2& s2) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << s1.logLikelihood;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s1.logPrior;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s1.logProposal;
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << s2.logLikelihood;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s2.logPrior;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << s2.logProposal;
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
