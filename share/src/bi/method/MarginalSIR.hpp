/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MARGINALSIR_HPP
#define BI_METHOD_MARGINALSIR_HPP

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
   * @name High-level interface.
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
   * @name Low-level interface.
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
   * @param s State.
   * @param inInit Initialisation file.
   *
   * @return Log-evidence.
   */
  template<class S1, class IO2>
  double init(Random& rng, const ScheduleIterator first, S1& s, IO2& inInit);

  /**
   * Step \f$x\f$-particles forward.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[out] s State.
   *
   * @return Log-evidence.
   */
  template<class S1>
  double step(Random& rng, const ScheduleIterator first,
      ScheduleIterator& iter, const ScheduleIterator last, S1& s);

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
   *
   * @return Acceptance rate.
   */
  template<class S1>
  double rejuvenate(Random& rng, const ScheduleIterator first,
      const ScheduleIterator now, S1& s);

  /**
   * Output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output(const S1& s, IO1& out);

  /**
   * Report progress on stderr.
   *
   * @param now Current step in time schedule.
   * @param ess Effective sample size of theta-particles.
   * @param r Was resampling performed?
   * @param acceptRate Acceptance rate of rejuvenation step (if any).
   */
  void report(const ScheduleElement now, const double ess, const bool r,
      const double acceptRate);

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
};
}

template<class B, class F, class A, class R>
bi::MarginalSIR<B,F,A,R>::MarginalSIR(B& m, F& mmh, A& adapter, R& resam,
    const int Nmoves) :
    m(m), mmh(mmh), adapter(adapter), resam(resam), Nmoves(Nmoves) {
  //
}

template<class B, class F, class A, class R>
template<class S1, class IO1, class IO2>
void bi::MarginalSIR<B,F,A,R>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  ScheduleIterator iter = first;
  s.les(iter->indexOutput()) = init(rng, iter, s, inInit);
  while (iter + 1 != last) {
    s.les(iter->indexOutput()) = step(rng, first, iter, last, s);
  }
  output(s, out);
  term();
}

template<class B, class F, class A, class R>
template<class S1, class IO2>
double bi::MarginalSIR<B,F,A,R>::init(Random& rng,
    const ScheduleIterator first, S1& s, IO2& inInit) {
  for (int p = 0; p < s.size(); ++p) {
    mmh.init(rng, first, first + 1, *s.thetas[p], inInit);
    s.logWeights()(p) = s.thetas[p]->logLikelihood;
    s.ancestors()(p) = p;
  }
  double le = logsumexp_reduce(s.logWeights())
      - bi::log(static_cast<double>(s.size()));

  return le;
}

template<class B, class F, class A, class R>
template<class S1>
double bi::MarginalSIR<B,F,A,R>::step(Random& rng,
    const ScheduleIterator first, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s) {
  int p, r;
  double acceptRate = 0.0, ess = 0.0;

  ess = resam.ess(s.logWeights());
  r = iter->isObserved() && resam.isTriggered(s.logWeights());
  if (r) {
    resample(rng, *iter, s);
    acceptRate = rejuvenate(rng, first, iter + 1, s);
  } else {
    Resampler::normalise(s.logWeights());
  }
  report(*iter, ess, r, acceptRate);

  ScheduleIterator iter1;
  for (p = 0; p < s.size(); ++p) {
    iter1 = iter;
    s.logWeights()(p) += mmh.extend(rng, iter1, last, *s.thetas[p]);
  }
  iter = iter1;

  double le = logsumexp_reduce(s.logWeights())
      - bi::log(static_cast<double>(s.size()));

  return le;
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::resample(Random& rng,
    const ScheduleElement now, S1& s) {
  if (now.isObserved()) {
    resam.resample(rng, s.logWeights(), s.ancestors(), s.thetas);
  }
}

template<class B, class F, class A, class R>
template<class S1>
double bi::MarginalSIR<B,F,A,R>::rejuvenate(Random& rng,
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

  return static_cast<double>(naccept) / totalMoves;
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::output(const S1& s, IO1& out) {
  out.write(s);
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::report(const ScheduleElement now,
    const double ess, const bool r, const double acceptRate) {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
#else
  const int rank = 0;
#endif

  if (rank == 0) {
    std::cerr << now.indexOutput() << ":\ttime " << now.getTime() << "\tESS "
        << ess;
    if (r) {
      std::cerr << "\tresample-move with acceptance rate " << acceptRate;
    }
    std::cerr << std::endl;
  }
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::term() {
  //
}

#endif
