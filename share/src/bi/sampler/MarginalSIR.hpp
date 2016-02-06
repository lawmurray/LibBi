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

#include "../state/Schedule.hpp"
#include "../misc/exception.hpp"
#include "../misc/TicToc.hpp"
#include "../primitive/vector_primitive.hpp"

#include <fstream>
#include <sstream>

namespace bi {
/**
 * Marginal sequential importance resampling.
 *
 * @ingroup method_sampler
 *
 * @tparam B Model type
 * @tparam F Filter type.
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
   * @param filter Filter.
   * @param adapter Adapter.
   * @param resam Resampler for theta-particles.
   * @param nmoves Number of move steps per \f$\theta\f$-particle after each
   * resample.
   * @param tmoves Total real time allocated to move steps.
   */
  MarginalSIR(B& m, F& filter, A& adapter, R& resam, const int nmoves = 1,
      const double tmoves = 0.0);

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
   * Adapt proposal.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] s State.
   */
  template<class S1>
  void adapt(const S1& s);

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
   * Move \f$\theta\f$-particles.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void move(Random& rng, const ScheduleIterator first,
      const ScheduleIterator iter, const ScheduleIterator now, S1& s);

  /**
   * @copydoc Simulator::outputT()
   */
  template<class S1, class IO1>
  void outputT(const S1& s, IO1& out);

  /**
   * Report progress on stderr.
   *
   * @param now Current step in time schedule.
   * @param s State.
   */
  template<class S1>
  void report(const ScheduleElement now, S1& s);

  /**
   * Report last step on stderr.
   *
   * @param now Current step in time schedule.
   */
  void reportT(const ScheduleElement now);

  /**
   * Finalise.
   */
  template<class S1>
  void term(Random& rng, S1& s);
  //@}

private:
  /**
   * Start of end of step for instrumentation.
   */
  enum StartOrEnd {
    START, END
  };

  /**
   * Step for instrumentation.
   */
  enum Step {
    INIT, ADAPT, RESAMPLE, MOVE, STEP, TERM
  };

  /**
   * Profiling output.
   */
  void profile(const StartOrEnd startOrEnd, const Step step);

#if ENABLE_DIAGNOSTICS == 4
  /**
   * Log file.
   */
  std::ofstream logFile;
#endif

  /**
   * Model.
   */
  B& m;

  /**
   * Filter.
   */
  F& filter;

  /**
   * Adapter.
   */
  A& adapter;

  /**
   * Resampler for the theta-particles
   */
  R& resam;

  /**
   * Clock.
   */
  TicToc clock;

  /**
   * Number of PMMH steps when moving.
   */
  int nmoves;

  /**
   * Total real time allocated to move steps.
   */
  double tmoves;

  /**
   * Was a resample performed on the last step?
   */
  bool lastResample;

  /**
   * Is the adapter ready?
   */
  bool adapterReady;

  /**
   * Last number of acceptances when move.
   */
  int lastAccept;

  /**
   * Last total number of moves.
   */
  int lastTotal;
};
}

template<class B, class F, class A, class R>
bi::MarginalSIR<B,F,A,R>::MarginalSIR(B& m, F& filter, A& adapter, R& resam,
    const int nmoves, const double tmoves) :
    m(m), filter(filter), adapter(adapter), resam(resam), nmoves(nmoves), tmoves(
        tmoves), lastResample(false), adapterReady(false), lastAccept(0), lastTotal(
        0) {
#if ENABLE_DIAGNOSTICS == 4
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
#else
  const int rank = 0;
  const int size = 1;
#endif
  std::stringstream buf;
  buf << "sir.log";
  if (size > 1) {
    buf << "." << rank;
  }
  logFile.open(buf.str().c_str());
#endif

  if (tmoves > 0.0) {
    /* always resample */
    resam.setEssRel(1.0);

    /* one move at a time */
    this->nmoves = 1;
  }
}

template<class B, class F, class A, class R>
template<class S1, class IO1, class IO2>
void bi::MarginalSIR<B,F,A,R>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  // should look very similar to Filter::filter()
  ScheduleIterator iter = first;
  profile(START, INIT);
  init(rng, iter, s, out, inInit);
  profile(END, INIT);
#if ENABLE_DIAGNOSTICS == 3
  std::stringstream buf;
  buf << "sir" << iter->indexOutput() << ".nc";
  SMCBuffer<SMCCache<ON_HOST,SMCNetCDFBuffer> > outtmp(m, s.size(), last->indexOutput(), buf.str(), REPLACE);
  outtmp.write(s);
  outtmp.flush();
#endif
  while (iter + 1 != last) {
    profile(START, ADAPT);
    adapt(s);
    profile(END, ADAPT);
    profile(START, RESAMPLE);
    resample(rng, *iter, s);
    profile(END, RESAMPLE);
    profile(START, MOVE);
    move(rng, first, iter, last, s);
    profile(END, MOVE);
    report(*iter, s);
    profile(START, STEP);
    step(rng, first, iter, last, s, out);
    profile(END, STEP);
#if ENABLE_DIAGNOSTICS == 3
    std::stringstream buf;
    buf << "sir" << iter->indexOutput() << ".nc";
    SMCBuffer<SMCCache<ON_HOST,SMCNetCDFBuffer> > outtmp(m, s.size(), last->indexOutput(), buf.str(), REPLACE);
    outtmp.write(s);
    outtmp.flush();
#endif
  }
  profile(START, TERM);
  term(rng, s);
  profile(END, TERM);
  reportT(*iter);
  outputT(s, out);
}

template<class B, class F, class A, class R>
template<class S1, class IO1, class IO2>
void bi::MarginalSIR<B,F,A,R>::init(Random& rng, const ScheduleIterator first,
    S1& s, IO1& out, IO2& inInit) {
  for (int p = 0; p < s.size(); ++p) {
    BOOST_AUTO(&s1, *s.s1s[p]);
    BOOST_AUTO(&out1, *s.out1s[p]);

    filter.init(rng, *first, s1, out1, inInit);
    filter.output0(s1, out1);
    filter.correct(rng, *first, s1);
    filter.output(*first, s1, out1);

    s.logWeights()(p) = s1.logLikelihood;
    s.ancestors()(p) = p;
  }

  double lW;
  s.ess = resam.reduce(s.logWeights(), &lW);
  s.logLikelihood = lW;
  s.logIncrements(0) = lW;

  out.clear();

  lastResample = false;
  adapterReady = false;
  lastAccept = 0;
  lastTotal = 0;
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::step(Random& rng, const ScheduleIterator first,
    ScheduleIterator& iter, const ScheduleIterator last, S1& s, IO1& out) {
  ScheduleIterator iter1;
  do {
    for (int p = 0; p < s.size(); ++p) {
      BOOST_AUTO(&s1, *s.s1s[p]);
      BOOST_AUTO(&out1, *s.out1s[p]);

      iter1 = iter;
      filter.step(rng, iter1, last, s1, out1);
      s.logWeights()(p) += s1.logIncrements(iter1->indexObs());
    }
    iter = iter1;
  } while (iter + 1 != last && !iter->isObserved());
#if ENABLE_DIAGNOSTICS == 3
  filter.samplePath(rng, s1, out1);
#endif

  double lW;
  s.ess = resam.reduce(s.logWeights(), &lW);
  s.logIncrements(iter->indexObs()) = lW - s.logLikelihood;
  s.logLikelihood = lW;
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::adapt(const S1& s) {
  adapterReady = adapter.ready();
  if (adapterReady) {
    adapter.clear();
    for (int p = 0; p < s.size(); ++p) {
      adapter.add(*s.s1s[p]);
    }
    adapter.adapt();
  }
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::resample(Random& rng,
    const ScheduleElement now, S1& s) {
  lastResample = resam.resample(rng, now, s);
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::move(Random& rng, const ScheduleIterator first,
    const ScheduleIterator iter, const ScheduleIterator last, S1& s) {
  if (lastResample) {
    /* compute budget */
    double milestone = tmoves * 1.0e6
        * bi::pow((iter + 2)->getTo() - first->getFrom(), 1.8)
        / bi::pow(last->getTo() - first->getFrom(), 1.8);
    int naccept = 0;
    int ntotal = 0;
    int p = 0;
    bool accept = false;
    bool complete = (tmoves <= 0.0 && p >= s.size())
        || (tmoves > 0.0 && clock.toc() >= milestone);

    while (!complete) {
      BOOST_AUTO(&s1, *s.s1s[p % s.size()]);
      BOOST_AUTO(&out1, *s.out1s[p % s.size()]);
      BOOST_AUTO(&s2, s.s2);
      BOOST_AUTO(&out2, s.out2);

      for (int move = 0; move < nmoves; ++move) {
        /* propose replacement */
        try {
          if (adapterReady) {
            filter.propose(rng, *first, s1, s2, out2, adapter);
          } else {
            filter.propose(rng, *first, s1, s2, out2);
          }
          if (bi::is_finite(s2.logPrior)) {
            filter.filter(rng, first, iter + 1, s2, out2);
          }
        } catch (CholeskyException e) {
          s2.logLikelihood = -BI_INF;
        } catch (ParticleFilterDegeneratedException e) {
          s2.logLikelihood = -BI_INF;
        }

        /* accept or reject */
        if (!bi::is_finite(s2.logLikelihood)) {
          accept = false;
        } else if (!bi::is_finite(s1.logLikelihood)) {
          accept = true;
        } else {
          double loglr = s2.logLikelihood - s1.logLikelihood;
          double logpr = s2.logPrior - s1.logPrior;
          double logqr = s1.logProposal - s2.logProposal;

          if (!bi::is_finite(s1.logProposal)
              && !bi::is_finite(s2.logProposal)) {
            logqr = 0.0;
          }
          double logratio = loglr + logpr + logqr;
          double u = rng.uniform<double>();

          accept = bi::log(u) < logratio;
        }

        if (accept) {
#if ENABLE_DIAGNOSTICS == 3
          filter.samplePath(rng, s2, out2);
#endif
          s1.swap(s2);
          out1.swap(out2);
          ++naccept;
        }
      }
      ++p;
      ++ntotal;
      complete = (tmoves <= 0.0 && p >= s.size())
          || (tmoves > 0.0 && clock.toc() >= milestone);
    }

    if (tmoves > 0.0) {
      /* particle moving when time elapsed must be eliminated */
      s.logWeights()(p % s.size()) = -BI_INF;
    }

    lastAccept = naccept;
    lastTotal = ntotal;
  }
}

template<class B, class F, class A, class R>
template<class S1, class IO1>
void bi::MarginalSIR<B,F,A,R>::outputT(const S1& s, IO1& out) {
  out.write(s);
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::report(const ScheduleElement now, S1& s) {
  if (lastResample) {
    int rank = 0, naccept = lastAccept, ntotal = lastTotal;

#ifdef ENABLE_MPI
    boost::mpi::communicator world;
    rank = world.rank();
    boost::mpi::reduce(world, lastAccept, naccept, std::plus<int>(), 0);
    boost::mpi::reduce(world, lastTotal, ntotal, std::plus<int>(), 0);
#endif
    if (rank == 0) {
      std::cerr << "\tmoves " << ntotal << "\taccepts " << naccept
      std::cerr << now.indexOutput() << ":\ttime " << now.getTime()
          << "\tESS " << s.ess;
      << "\trate " << double(naccept) / ntotal;
      std::cerr << std::endl;
    }
  }
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::reportT(const ScheduleElement now) {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
#else
  const int rank = 0;
#endif

  if (rank == 0) {
    std::cerr << now.indexOutput() << ":\ttime " << now.getTime()
        << "\t...finished." << std::endl;
  }
}

template<class B, class F, class A, class R>
template<class S1>
void bi::MarginalSIR<B,F,A,R>::term(Random& rng, S1& s) {
  s.logLikelihood += logsumexp_reduce(s.logWeights())
      - bi::log(double(s.size()));
  for (int p = 0; p < s.size(); ++p) {
    BOOST_AUTO(&s1, *s.s1s[p]);
    BOOST_AUTO(&out1, *s.out1s[p]);
    filter.samplePath(rng, s1, out1);
  }
}

template<class B, class F, class A, class R>
void bi::MarginalSIR<B,F,A,R>::profile(const StartOrEnd startOrEnd,
    const Step step) {
#if ENABLE_DIAGNOSTICS == 4
  if (startOrEnd == START) {
    clock.sync();
  }
#endif
  if (startOrEnd == START && step == INIT) {
    clock.tic();
  }
#if ENABLE_DIAGNOSTICS == 4
  if (startOrEnd == START) {
    logFile << step << ',' << clock.toc();
  } else {
    logFile << ',' << clock.toc() << std::endl;
  }
#endif
}

#endif
