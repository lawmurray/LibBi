/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_PARTICLEMARGINALMETROPOLISHASTINGS_HPP
#define BI_METHOD_PARTICLEMARGINALMETROPOLISHASTINGS_HPP

#include "misc.hpp"
#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/view.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"
#include "AdaptiveNParticleFilter.hpp"

#include "boost/serialization/serialization.hpp"

namespace bi {
/**
 * Particle filtering modes for ParticleMarginalMetropolisHastings.
 */
enum FilterMode {
  /**
   * Unconditioned filter.
   */
  UNCONDITIONED,

  /**
   * Filter conditioned on current state trajectory.
   */
  CONDITIONED
};

/**
 * %State of ParticleMarginalMetropolisHastings.
 */
class ParticleMarginalMetropolisHastingsState {
public:
  typedef host_vector<real, pinned_allocator<real> > vector_type;
  typedef host_matrix<real, pinned_allocator<real> > matrix_type;

  /**
   * Constructor.
   *
   * @tparam B Model type.
   *
   * @param m Model.
   * @param M Size of state.
   * @param T Number of time points.
   */
  template<class B>
  ParticleMarginalMetropolisHastingsState(B& m, const int M, const int T = 0);

  /**
   * Copy constructor.
   */
  ParticleMarginalMetropolisHastingsState(const ParticleMarginalMetropolisHastingsState& o);

  /**
   * Assignment.
   */
  ParticleMarginalMetropolisHastingsState& operator=(const ParticleMarginalMetropolisHastingsState& o);

  /**
   * Resize.
   *
   * @param T Number of time points.
   */
  void resize(const int T);

  /**
   * Swap.
   */
  void swap(ParticleMarginalMetropolisHastingsState& o);

  /**
   * State of p-vars.
   */
  vector_type theta;

  /**
   * Trajectory of d-vars. Rows index variables, columns times.
   */
  matrix_type xd;

  /**
   * Trajectory of r-vars. Rows index variables, columns times.
   */
  matrix_type xr;

  /**
   * Marginal log-likelihood of parameters.
   */
  real ll;

  /**
   * Log-prior density of parameters.
   */
  real lp;

  /**
   * Log-proposal density of parameters.
   */
  real lq;

private:
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
};
}

template<class B>
inline bi::ParticleMarginalMetropolisHastingsState::ParticleMarginalMetropolisHastingsState(B& m, const int M,
    const int T) :
    theta(M),
    xd(m.getNetSize(D_VAR), T),
    xr(m.getNetSize(R_VAR), T),
    ll(-1.0/0.0),
    lp(-1.0/0.0),
    lq(-1.0/0.0) {
  //
}

inline bi::ParticleMarginalMetropolisHastingsState::ParticleMarginalMetropolisHastingsState(const ParticleMarginalMetropolisHastingsState& o) :
        theta(o.theta.size()),
        xd(o.xd.size1(), o.xd.size2()),
        xr(o.xr.size1(), o.xr.size2()) {
  operator=(o); // ensures deep copy
}

inline bi::ParticleMarginalMetropolisHastingsState& bi::ParticleMarginalMetropolisHastingsState::operator=(
    const ParticleMarginalMetropolisHastingsState& o) {
  theta = o.theta;
  xd = o.xd;
  xr = o.xr;
  ll = o.ll;
  lp = o.lp;
  lq = o.lq;

  return *this;
}

inline void bi::ParticleMarginalMetropolisHastingsState::resize(const int T) {
  xd.resize(xd.size1(), T);
  xr.resize(xr.size1(), T);
}

inline void bi::ParticleMarginalMetropolisHastingsState::swap(ParticleMarginalMetropolisHastingsState& o) {
  theta.swap(o.theta);
  xd.swap(o.xd);
  xr.swap(o.xr);
  std::swap(ll, o.ll);
  std::swap(lp, o.lp);
  std::swap(lq, o.lq);
}

template<class Archive>
inline void bi::ParticleMarginalMetropolisHastingsState::serialize(
    Archive& ar, const unsigned version) {
  ar & theta;
  ar & xd;
  ar & xr;
  ar & ll;
  ar & lp;
  ar & lq;
}

namespace bi {
/**
 * Particle Marginal Metropolis-Hastings (PMMH) sampler.
 *
 * See @ref Jones2010 "Jones, Parslow & Murray (2009)", @ref Andrieu2010
 * "Andrieu, Doucet \& Holenstein (2010)". Adaptation is supported according
 * to @ref Haario2001 "Haario, Saksman & Tamminen (2001)".
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam IO1 #concept::ParticleMarginalMetropolisHastingsBuffer type.
 * @tparam CL Cache location.
 */
template<class B, class IO1, Location CL = ON_HOST>
class ParticleMarginalMetropolisHastings {
public:
  /**
   * State type.
   */
  typedef ParticleMarginalMetropolisHastingsState state_type;

  /**
   * Constructor.
   *
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param m Model.
   * @param out Output.
   *
   * @see ParticleFilter
   */
  ParticleMarginalMetropolisHastings(B& m, IO1* out = NULL);

  /**
   * Get output buffer.
   */
  IO1* getOutput();

  /**
   * Get current state of chain.
   *
   * @return Current state of chain.
   */
  ParticleMarginalMetropolisHastingsState& getState();

  /**
   * Get the last state compared to the current state. If the last step was
   * accepted, this will be the previous state of the chain. If the last
   * step was rejected, this will be the last proposal.
   *
   * @return Last state compared to current state.
   */
  ParticleMarginalMetropolisHastingsState& getOtherState();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Length of time to sample over.
   * @param s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   * @param C Number of samples to draw.
   */
  template<Location L, class F, class IO2>
  void sample(Random& rng, const real T, State<B,L>& s, F* filter,
      IO2* inInit = NULL, const int C = 1, const FilterMode = UNCONDITIONED);
  //@}
  template<Location L, class F, class IO2>
  void sample_together(Random& rng, const real T, State<B,L>& s, F* filter, IO2* inInit, const int C);

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
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   */
  template<Location L, class F, class IO2>
  void init(Random& rng, State<B,L>& s, F* filter, IO2* inInit = NULL);

  /**
   * Take one step.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   *
   * @param rng Random number generator.
   * @param T Length of time to sample over.
   * @param s State.
   * @param filter Filter.
   * @param type Type of filtering to perform.
   *
   * @return True if the step is accepted, false otherwise.
   */
  template<Location L, class F>
  bool step(Random& rng, const real T, State<B,L>& s, F* filter,
      const FilterMode type = UNCONDITIONED);

  template<Location L>
  void propose(Random& rng, State<B,L>& s);
  template<Location L>
  void computePriorOnProposal(State<B,L>& s);
  template<Location L, class F>
  bool estimateLLOnProposal(Random& rng, const real T, State<B,L>& s, F* filter,
      const FilterMode filtermode = UNCONDITIONED);
  template<class F>
  void computeAcceptReject(Random& rng, F* filter, bool cpf_fail, bool & result);
  template<bi::Location L, class F>
  real computeLROnProposal(Random& rng, const real T, State<B,L>& s,
      State<B,L>& s_2, F* filter);
  void computeAcceptReject(Random& rng, bool cpf_fail, bool & result);
  template<bi::Location L, class F>
  bool step_together(Random& rng, const real T, State<B,L>& s, F* filter);

  /**
   * Output current state.
   *
   * @param c Index in output file.
   */
  void output(const int c);

  /**
   * Report progress on stderr.
   *
   * @param c Number of steps taken.
   */
  void report(const int c);

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
  //@}

private:
  /**
   * Accept step.
   *
   * @tparam #concept::Filter type.
   *
   * @param rng Random number generator.
   * @param filter Filter.
   */
  template<class F>
  void accept(Random& rng, F* filter);

  /**
   * Reject step.
   */
  void reject();

  /**
   * Model.
   */
  B& m;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Current state.
   */
  ParticleMarginalMetropolisHastingsState x1;

  /**
   * Previous or proposed state.
   */
  ParticleMarginalMetropolisHastingsState x2;

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

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
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
template<Location CL = ON_HOST>
struct ParticleMarginalMetropolisHastingsFactory {
  /**
   * Create particle MCMC sampler.
   *
   * @return ParticleMarginalMetropolisHastings object. Caller has ownership.
   *
   * @see ParticleMarginalMetropolisHastings::ParticleMarginalMetropolisHastings()
   */
  template<class B, class IO1>
  static ParticleMarginalMetropolisHastings<B,IO1,CL>* create(B& m,
      IO1* out = NULL) {
    return new ParticleMarginalMetropolisHastings<B,IO1,CL>(m, out);
  }
};
}

#include "../math/misc.hpp"

template<class B, class IO1, bi::Location CL>
bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::ParticleMarginalMetropolisHastings(
    B& m, IO1* out) :
    m(m),
    out(out),
    x1(m, NP, (out != NULL) ? out->size2() : 0),
    x2(m, NP, (out != NULL) ? out->size2() : 0),
    lastAccepted(false),
    accepted(0),
    total(0) {
  //
}

template<class B, class IO1, bi::Location CL>
IO1* bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL>
bi::ParticleMarginalMetropolisHastingsState& bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::getState() {
  return x1;
}

template<class B, class IO1, bi::Location CL>
bi::ParticleMarginalMetropolisHastingsState& bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::getOtherState() {
  return x2;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::sample(Random& rng,
    const real T, State<B,L>& s, F* filter, IO2* inInit, const int C, const FilterMode filterMode) {
  int c;
  const int P = s.size();
  init(rng, s, filter, inInit);
  for (c = 0; c < C; ++c) {
    step(rng, T, s, filter, filterMode);
    //    if (c % 1000 == 0) {
    report(c);
    //    }
    output(c);
    s.setRange(0,P);
  }
  term();
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::sample_together(Random& rng,
    const real T, State<B,L>& s, F* filter, IO2* inInit, const int C) {
  int c;
  const int P = s.size();
  init(rng, s, filter, inInit);

  step(rng, T, s, filter, UNCONDITIONED);
  report(0);
  output(0);
  s.setRange(0,P);

  for (c = 1; c < C; ++c) {
    step_together(rng, T, s, filter);
//    if (c % 1000 == 0) {
      report(c);
//    }
    output(c);
    s.setRange(0,P);
  }
  term();
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::init(Random& rng,
    State<B,L>& s, F* filter, IO2* inInit) {
  /* first run of filter to get everything initialised properly */
  filter->rewind();
  filter->filter(rng, 0.0, s, inInit);

  /* initialise state vector */
  subrange(x1.theta, 0, NP) = vec(s.get(P_VAR));
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::propose(Random& rng,
    State<B,L>& s) {
  typename loc_temp_vector<L,real>::type lp(1);
  const int P = s.size();
  s.setRange(0, 1);
  /* proposal */
  row(s.get(P_VAR), 0) = subrange(x1.theta, 0, NP);
  m.proposalParameterSamples(rng, s);
  subrange(x2.theta, 0, NP) = row(s.get(P_VAR), 0);

  /* reverse proposal log-density */
  lp.clear();
  row(s.get(P_VAR), 0) = subrange(x2.theta, 0, NP);
  row(s.getAlt(P_VAR), 0) = subrange(x1.theta, 0, NP);
  m.proposalParameterLogDensities(s, lp);
  x1.lq = *lp.begin();

  /* proposal log-density */
  lp.clear();
  row(s.get(P_VAR), 0) = subrange(x1.theta, 0, NP);
  row(s.getAlt(P_VAR), 0) = subrange(x2.theta, 0, NP);
  m.proposalParameterLogDensities(s, lp);
  x2.lq = *lp.begin();
  s.setRange(0, P);
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::computePriorOnProposal(State<B,L>& s) {

  const int P = s.size();
  s.setRange(0, 1);
  /* prior log-density */
  typename loc_temp_vector<L,real>::type lp(1);
  lp.clear();
  row(s.getAlt(P_VAR), 0) = subrange(x2.theta, 0, NP);
  m.parameterLogDensities(s, lp);
  x2.lp = *lp.begin();
  s.setRange(0, P);
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
bool bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::estimateLLOnProposal(Random& rng,
    const real T, State<B,L>& s, F* filter, const FilterMode filtermode) {
  /* prepare for filter */
  bool cpf_fail = false;

  m.initialSamples(rng, s);

  /* likelihood */
  filter->rewind();
  try {
    x2.ll = filter->filter(rng, T, x2.theta, s);
  } catch (CholeskyException e) {
    x2.ll = -1.0/0.0; // -inf
  } catch (ParticleFilterDegeneratedException e) {
    x2.ll = -1.0/0.0; // -inf
  }

  /* check for x1.lp is 0 to avoid computation of ll at step 1
   this is to prevent numerical instability associated with a
   (virtual) trajectory with no weight. */
  if (filtermode == CONDITIONED && x1.lp > 0) {
    m.initialSamples(rng, s);
    filter->rewind();
    try {
      x1.ll = filter->filter(rng, T, x1.theta, s, x1.xd, x1.xr);
    } catch (CholeskyException e) {
      x1.ll = 1.0/0.0; // +inf
      cpf_fail = true;
    } catch (ParticleFilterDegeneratedException e) {
      x1.ll = 1.0/0.0; // +inf
      cpf_fail = true;
    }
  }
  return cpf_fail;
}

template<class B, class IO1, bi::Location CL>
template<class F>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::computeAcceptReject(Random& rng,
    F* filter, bool cpf_fail, bool & result){
  /* accept/reject */
  if (!result || !BI_IS_FINITE(x2.ll) || cpf_fail) {
    result = false;
  } else if (!BI_IS_FINITE(x1.ll)) {
    result = true;
  } else {
    real loglr = x2.ll - x1.ll;
    real logpr = x2.lp - x1.lp;
    real logqr = x1.lq - x2.lq;
    if (!is_finite(x1.lq) && !is_finite(x2.lq)) {
      logqr = 0.0;
    }
    real logratio = loglr + logpr + logqr;
    real u = rng.uniform<real>();

    result = std::log(u) < logratio;
  }
  if (result) {
    accept(rng, filter);
  } else {
    reject();
  }
}


template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
bool bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::step(Random& rng,
    const real T, State<B,L>& s, F* filter, const FilterMode filtermode) {
  /* pre-conditions */
  assert(filter->getOutput() != NULL);
  assert(out == NULL || filter->getOutput()->size2() == out->size2());
  // propose new parameter value
  // and compute the proposal density and its reverse
  propose(rng, s);
  // compute ths log prior density on the proposed
  // parameter value
  computePriorOnProposal(s);
  bool result = true;
  bool cpf_fail = false;
  if (!BI_IS_FINITE(x2.lp)) {
    result = false;
    x2.ll = 0;
  } else {
    // estimate log likelihood on proposed parameter value
    cpf_fail = estimateLLOnProposal(rng, T, s, filter, filtermode);
  }
  // compute acceptance ratio and draw uniform
  // the result is written in result
  computeAcceptReject(rng, filter, cpf_fail, result);
  return result;
}


template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
bool bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::step_together(Random& rng,
    const real T, State<B,L>& s, F* filter) {
  /* pre-conditions */
  assert(out == NULL || filter->getOutput()->size2() == out->size2());

  typename temp_host_vector<real>::type lp(1);

  State<B,L> s_2(m,s.size());
  s_2 = s;

  real loglr = 0.0;

  propose(rng, s);
  computePriorOnProposal(s);

  bool result = true;
  if (!BI_IS_FINITE(x2.lp)) {
    result = false;
    x2.ll = 0;
  } else {
    loglr = computeLROnProposal(rng, T, s, s_2, filter);
  }

  /* accept/reject */
  if (!result || !BI_IS_FINITE(x2.ll)) {
    result = false;
  } else {
    //    real loglr = x2.ll - x1.ll;
    real logpr = x2.lp - x1.lp;
    real logqr = x1.lq - x2.lq;
    if (!is_finite(x1.lq) && !is_finite(x2.lq)) {
      logqr = 0.0;
    }
    real logratio = loglr + logpr + logqr;
    real u = rng.uniform<real>();

    result = std::log(u) < logratio;
  }

  if (result) {
    accept(rng, filter);
    s = s_2;
  } else {
    reject();
  }

  return result;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
real bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::computeLROnProposal(Random& rng,
    const real T, State<B,L>& s, State<B,L>& s_2, F* filter) {
  /* prepare for filter */
  m.initialSamples(rng, s);
  m.initialSamples(rng, s_2);

  /* likelihood */
  filter->rewind();

  real ll1 = 0.0, ll2 = 0.0;
  real loglr = filter->filter(rng, T, x1.theta, s, x1.xd, x1.xr, x2.theta,
      s_2, ll1, ll2);
  x1.ll = ll1;
  x2.ll = ll2;
  return loglr;
}

template<class B, class IO1, bi::Location CL>
template<class F>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::accept(Random& rng,
    F* filter) {
  x1.swap(x2);
  ++accepted;
  ++total;
  lastAccepted = true;
  if (filter->getOutput() != NULL) {
    filter->sampleTrajectory(rng, x1.xd, x1.xr);
  }
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::reject() {
  ++total;
  lastAccepted = false;
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::output(const int c) {
  if (out != NULL) {
    out->writeSample(c, subrange(x1.theta, 0, NP)); // only p-var portion
    out->writeParticle(c, x1.xd, x1.xr);
    out->writeLogLikelihood(c, x1.ll);
    out->writeLogPrior(c, x1.lp);
  }
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::report(const int c) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << this->getState().ll;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getState().lp;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getState().lq;
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << this->getOtherState().ll;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getOtherState().lp;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getOtherState().lq;
  std::cerr << '\t';
  if (this->wasLastAccepted()) {
    std::cerr << "accept";
  }
  std::cerr << "\taccept=" << (double)accepted/total;
  std::cerr << std::endl;
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::term() {
  //
}

template<class B, class IO1, bi::Location CL>
inline bool bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::wasLastAccepted() {
  return lastAccepted;
}

template<class B, class IO1, bi::Location CL>
inline int bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::getNumSteps() {
  return total;
}

template<class B, class IO1, bi::Location CL>
inline int bi::ParticleMarginalMetropolisHastings<B,IO1,CL>::getNumAccepted() {
  return accepted;
}

#endif
