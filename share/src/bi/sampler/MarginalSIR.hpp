/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SAMPLER_MARGINALSIR_HPP
#define BI_SAMPLER_MARGINALSIR_HPP

#include "misc.hpp"
#include "../state/Schedule.hpp"
#include "../misc/exception.hpp"
#include "../primitive/vector_primitive.hpp"

namespace bi {
/**
 * Marginal sequential importance resampling.
 *
 * @ingroup method_sampler
 *
 * @tparam B Model type
 * @tparam F MarginalMH type.
 * @tparam A Adapter type.
 * @tparam R Resampler type.
 *
 * Implements sequential importance resampling over parameters, which, when
 * combined with a particle filter, gives the SMC^2 method described in
 * @ref Chopin2013 "Chopin, Jacob \& Papaspiliopoulos (2013)".
 *
 * @todo Add support for adapter classes.
 * @todo Add support for stopper classes for theta particles.
 */
template<class B, class F, class A, class R>
class MarginalSIR {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param mmh PMMH sampler.
   * @param adapter Adapter.
   * @param resam Resampler for theta-particles.
   * @param Nmoves Number of steps per \f$\theta\f$-particle.
   * @param adapter Proposal adaptation strategy.
   * @param adapterScale Scaling factor for local proposals.
   * @param out Output.
   */
  MarginalSIR(B& m, F& mmh, A& adapter, R& resam, const int Nmoves = 1);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc MarginalMH::sample()
   */
  template<class S1, class IO1, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, const int C, IO1& out, IO2& inInit);
  //@}

  /**
   * @name Low-level interface
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
   * @param s State.
   * @param out Output buffer.
   * @param inInit Init buffer.
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleIterator first, S1& s, IO1& out,
      IO2& inInit);

  /**
   * Step \f$x\f$-particles forward.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[out] s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void step(Random& rng, const ScheduleIterator first, ScheduleIterator& iter,
      const ScheduleIterator last, S1& s, IO1& out);

  /**
   * Resample \f$\theta\f$-particles.
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
   * Rejuvenate \f$\theta\f$-particles.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param now Current position in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void rejuvenate(Random& rng, const ScheduleIterator first,
      const ScheduleIterator now, S1& s);

  /**
   * @copydoc Simulator::output0()
   */
  template<class S1, class IO1>
  void output0(const S1& s, IO1& out);

  /**
   * @copydoc Simulator::output()
   */
  template<class S1, class IO1>
  void output(const ScheduleElement now, const S1& s, IO1& out);

  /**
   * @copydoc Simulator::outputT()
   */
  template<class S1, class IO1>
  void outputT(const S1& s, IO1& out);

  /**
   * Report progress on stderr.
   *
   * @param now Current step in time schedule.
   * @param ess Effective sample size of theta-particles.
   * @param r Was resampling performed?
   * @param acceptRate Acceptance rate of rejuvenation step (if any).
   */
  void report(const ScheduleElement now);

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
   * Marginal MH sampler.
   */
  F& mmh;

  /**
   * Adapter.
   */
  A& adapter;

  /**
   * Resampler for the theta-particles
   */
  R& resam;

  /**
   * Number of PMMH steps when rejuvenating.
   */
  int Nmoves;

  /**
   * Was a resample performed on the last step?
   */
  bool lastResample;

  /**
   * Last ESS when testing to resample.
   */
  double lastEss;

  /**
   * Last acceptance rate when rejuvenating.
   */
  double lastAcceptRate;
};
}

template<class B, class F, class A, class R>
bi::MarginalSIR<B,F,A,R>::MarginalSIR(B& m, F& mmh, A& adapter, R& resam,
    const int Nmoves) :
    m(m), mmh(mmh), adapter(adapter), resam(resam), Nmoves(Nmoves), lastResample(
        false), lastEss(0.0), lastAcceptRate(0.0) {
  //
}

template<class B, class F, class A, class R>
template<class S1, class IO1, class IO2>
void bi::MarginalSIR<B,F,A,R>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  // should be very similar to Filter::filter()
  ScheduleIterator iter = first;
  init(rng, *iter, s, out, inInit);
  output0(s, out);
  correct(rng, *iter, s);
  output(*iter, s, out);
  while (iter + 1 != last) {
    step(rng, first, iter, last, s, out);
  }
  term(s);
  output0(s, out);
}

template<class B, class F, class A, class R>
template<class S1, class IO1, class IO2>
void bi::MarginalSIR<B,F,A,R>::init(Random& rng, const ScheduleIterator first,
    S1& s, IO1& out, IO2& inInit) {
  for (int p = 0; p < s.size(); ++p) {
    mmh.init(rng, first, first + 1, *s.thetas[p], inInit);
    s.logWeights()(p) = s.thetas[p]->logLikelihood;
    s.ancestors()(p) = p;
  }
  out.clear();

  lastResample = false;
  lastEss = 0.0;
  lastAcceptRate = 0.0;
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::step(Random& rng, const ScheduleIterator first,
    ScheduleIterator& iter, const ScheduleIterator last, S1& s, IO1& out) {
  ScheduleIterator iter1;
  int p;

  do {
    resample(rng, *iter, s);
    rejuvenate(rng, first, iter + 1, s);
    report(*iter);

    ++iter;
    for (p = 0; p < s.size(); ++p) {
      iter1 = iter;
      s.logWeights()(p) += mmh.extend(rng, iter1, last, *s.thetas[p]);
    }
    iter = iter1;
    output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::resample(Random& rng,
    const ScheduleElement now, S1& s) {
  double lW;
  lastResample = false;
  if (resam.isTriggered(now, s, &lW, &lastEss)) {
    lastResample = true;
    if (resampler_needs_max < R > ::value && now.isObserved()) {
      resam.setMaxLogWeight(getMaxLogWeight(now, s));
    }
    typename precompute_type<R,S1::location>::type pre;
    resam.precompute(s.logWeights(), pre);
    if (now.hasOutput()) {
      resam.ancestorsPermute(rng, s.logWeights(), s.ancestors(), pre);
      resam.copy(s.ancestors(), s.thetas);
    } else {
      typename S1::temp_int_vector_type as1(s.ancestors().size());
      resam.ancestorsPermute(rng, s.logWeights(), as1, pre);
      resam.copy(as1, s.thetas);
      bi::gather(as1, s.ancestors(), s.ancestors());
    }
    s.logWeights().clear();
    s.logLikelihood += lW;
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::rejuvenate(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s) {
  int p, move, naccept = 0;
  for (p = 0; p < s.size(); ++p) {
    for (move = 0; move < Nmoves; ++move) {
      mmh.propose(rng, first, last, *s.thetas[p], s.theta2);
      if (mmh.acceptReject(rng, *s.thetas[p], s.theta2)) {
        ++naccept;
      }
    }
  }

  int totalMoves = Nmoves * s.size();
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
  boost::mpi::all_reduce(world, &totalMoves, 1, &totalMoves, std::plus<int>());
  boost::mpi::all_reduce(world, &naccept, 1, &naccept, std::plus<int>());
#endif
  lastAcceptRate = static_cast<double>(naccept) / totalMoves;
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::output0(const S1& s, IO1& out) {
  //
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::output(const ScheduleElement now, const S1& s,
    IO1& out) {
  //
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::outputT(const S1& s, IO1& out) {
  out.write(s);
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::report(const ScheduleElement now) {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
#else
  const int rank = 0;
#endif

  if (rank == 0) {
    std::cerr << now.indexOutput() << ":\ttime " << now.getTime() << "\tESS "
        << lastEss;
    if (lastResample) {
      std::cerr << "\tresample-move with acceptance rate " << lastAcceptRate;
    }
    std::cerr << std::endl;
  }
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::term() {
  //
}

#endif
