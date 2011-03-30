/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_PARTICLEMCMC_HPP
#define BI_METHOD_PARTICLEMCMC_HPP

#include "StratifiedResampler.hpp"
#include "misc.hpp"
#include "../state/Static.hpp"
#include "../state/State.hpp"
#include "../math/host_vector.hpp"
#include "../math/host_matrix.hpp"
#include "../math/locatable.hpp"
#include "../misc/pinned_allocator.hpp"
#include "../misc/TicToc.hpp"
#include "../pdf/FactoredPdf.hpp"
#include "../typelist/easy_typelist.hpp"

#ifndef __CUDACC__
#include "boost/serialization/serialization.hpp"
#endif

namespace bi {
/**
 * @internal
 *
 * %State of ParticleMCMC.
 */
class ParticleMCMCState {
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
  ParticleMCMCState(B& m, const int M, const int T = 0);

  /**
   * Copy constructor.
   */
  ParticleMCMCState(const ParticleMCMCState& o);

  /**
   * Assignment.
   */
  ParticleMCMCState& operator=(const ParticleMCMCState& o);

  /**
   * Resize.
   *
   * @param T Number of time points.
   */
  void resize(const int T);

  /**
   * Swap.
   */
  void swap(ParticleMCMCState& o);

  /**
   * State of p-nodes.
   */
  vector_type theta;

  /**
   * State of s-nodes.
   */
  vector_type xs;

  /**
   * Trajectory of d-node. Rows index variables, columns times.
   */
  matrix_type xd;

  /**
   * Trajectory of c-nodes. Rows index
   * variables, columns times.
   */
  matrix_type xc;

  /**
   * Trajectory of r-nodes. Rows index variables, columns times.
   */
  matrix_type xr;

  /**
   * Log-likelihood contribution at each time.
   */
  vector_type lls;

  /**
   * Effective sample sizes at each time.
   */
  vector_type ess;

  /**
   * Marginal log-likelihood of parameters.
   */
  real ll;

  /**
   * Log-prior density of parameters.
   */
  real lp;

  /**
   * Proposal density of parameters.
   */
  real lq;

  /**
   * TimeStamp time in computing likelihood.
   */
  int timeStamp;

private:
  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};
}

template<class B>
inline bi::ParticleMCMCState::ParticleMCMCState(B& m, const int M,
    const int T) :
    theta(M),
    xs(m.getNetSize(S_NODE)),
    xd(m.getNetSize(D_NODE), T),
    xc(m.getNetSize(C_NODE), T),
    xr(m.getNetSize(R_NODE), T),
    lls(T),
    ess(T),
    ll(0.0),
    lp(0.0),
    lq(0.0),
    timeStamp(0) {
  //
}

inline bi::ParticleMCMCState::ParticleMCMCState(const ParticleMCMCState& o) :
    theta(o.theta.size()),
    xs(o.xs.size()),
    xd(o.xd.size1(), o.xd.size2()),
    xc(o.xc.size1(), o.xc.size2()),
    xr(o.xr.size1(), o.xr.size2()),
    lls(o.lls.size()),
    ess(o.ess.size()) {
  operator=(o); // ensures deep copy
}

inline bi::ParticleMCMCState& bi::ParticleMCMCState::operator=(
    const ParticleMCMCState& o) {
  theta = o.theta;
  xs = o.xs;
  xd = o.xd;
  xc = o.xc;
  xr = o.xr;
  lls = o.lls;
  ess = o.ess;
  ll = o.ll;
  lp = o.lp;
  lq = o.lq;
  timeStamp = o.timeStamp;

  return *this;
}

inline void bi::ParticleMCMCState::resize(const int T) {
  xd.resize(xd.size1(), T);
  xc.resize(xc.size1(), T);
  xr.resize(xr.size1(), T);
  lls.resize(T, true);
  ess.resize(T, true);
}

inline void bi::ParticleMCMCState::swap(ParticleMCMCState& o) {
  theta.swap(o.theta);
  std::swap(ll, o.ll);
  std::swap(lp, o.lp);
  std::swap(lq, o.lq);
  std::swap(timeStamp, o.timeStamp);
  xs.swap(o.xs);
  xd.swap(o.xd);
  xc.swap(o.xc);
  xr.swap(o.xr);
  lls.swap(o.lls);
  ess.swap(o.ess);
}

#ifndef __CUDACC__
template<class Archive>
inline void bi::ParticleMCMCState::serialize(Archive& ar, const unsigned version) {
  ar & theta;
  ar & xs;
  ar & xd;
  ar & xc;
  ar & xr;
  ar & lls;
  ar & ess;
  ar & ll;
  ar & lp;
  ar & lq;
  ar & timeStamp;
}
#endif

namespace bi {
/**
 * Particle Markov chain Monte Carlo sampler. Supports Particle Marginal
 * Metropolis-Hastings (PMMH) and Particle Gibbs if Gibbs update of
 * parameters is handled externally.
 *
 * See @ref Jones2010 "Jones, Parslow & Murray (2009)", @ref Andrieu2010
 * "Andrieu, Doucet \& Holenstein (2010)". Adaptation is supported according
 * to @ref Haario2001 "Haario, Saksman & Tamminen (2001)".
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam IO1 #concept::ParticleMCMCBuffer type.
 * @tparam CL Cache location.
 */
template<class B, class IO1, Location CL = ON_HOST>
class ParticleMCMC {
private:
  typedef typename B::prior_type factor_type;
  typedef typename push_back<empty_typelist,factor_type,3>::type factor_typelist;
public:
  /**
   * State type.
   */
  typedef ParticleMCMCState state_type;

  /**
   * Prior type.
   */
  typedef FactoredPdf<factor_typelist> prior_type;


  /**
   * Constructor.
   *
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param out Output.
   * @param flag Indicates how initial conditions should be handled.
   *
   * @see ParticleFilter
   */
  ParticleMCMC(B& m, Random& rng, IO1* out = NULL,
      const InitialConditionType flag = INITIAL_SAMPLED);

  /**
   * Get output buffer.
   */
  IO1* getOutput();

  /**
   * Get current state of chain.
   *
   * @return Current state of chain.
   */
  ParticleMCMCState& getState();

  /**
   * Get the last state compared to the current state. If the last step was
   * accepted, this will be the previous state of the chain. If the last
   * step was rejected, this will be the last proposal.
   *
   * @return Last state compared to current state.
   */
  ParticleMCMCState& getOtherState();

  /**
   * Get prior distribution.
   *
   * @return Prior distribution.
   */
  prior_type& getPrior();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Sample.
   *
   * @tparam Q1 Conditional pdf type.
   * @tparam V1 Vector type.
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam R #concept::Resampler type.
   *
   * @param q Proposal distribution.
   * @param x Initial state.
   * @param C Number of samples to draw.
   * @param T Length of time to sample over.
   * @param theta Static state.
   * @param s State.
   * @param filter Filter.
   * @param resampler Resampler.
   * @param sd Proposal adaptation scaling parameter. Zero gives default.
   * @param A Number of steps after which to start adaptation. Negative gives
   * no adaptation.
   *
   * Note that @c theta.get(P_NODE) should be initialised with the starting
   * state of the chain.
   */
  template<class Q1, class V1, Location L, class F, class R>
  void sample(Q1& q, const V1 x, const int C, const real T, Static<L>& theta,
      State<L>& s, F* filter, R* resampler = NULL, const real sd = 0.0,
      const int A = -1);
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
   * @tparam V1 Vector type.
   * @tparam L Location.
   * @tparam F Filter.
   * @tparam R Resampler.
   *
   * @param x Initial state.
   * @param T Length of time to sample over.
   * @param theta Static state.
   * @param s State.
   * @param filter Filter.
   * @param resampler Resampler.
   */
  template<class V1, Location L, class F, class R>
  void init(const V1 x, const real T, Static<L>& theta, State<L>& s,
      F* filter, R* resampler);
  /**
   * Propose new state of Markov chain.
   *
   * @tparam Q1 Conditional pdf type.
   *
   * @param q Proposal distribution.
   */
  template<class Q1>
  void propose(Q1& q);

  /**
   * Propose new state of Markov chain.
   *
   * @tparam V1 Vector type.
   *
   * @param theta State.
   *
   * This call should be followed by accept() or reject() to accept or reject
   * the proposed state, metropolis() or metropolisHastings() will not work,
   * as the source proposal distribution is not known in this case.
   */
  template<class V1>
  void proposal(const V1 theta);

  /**
   * Compute prior of proposed state of chain.
   */
  void prior();

  /**
   * Compute likelihood of proposed state of chain.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam R #concept::Resampler type.
   *
   * @param T Length of time to sample over.
   * @param theta Static state.
   * @param s State.
   * @param filter Filter.
   * @param resampler Resampler.
   * @param type Type of filtering to perform.
   */
  template<Location L, class F, class R>
  void likelihood(const real T, Static<L>& theta, State<L>& s,
      F* filter, R* resampler = NULL, const FilterType type = UNCONDITIONED);

  /**
   * Apply Metropolis-Hastings criterion to accept or reject proposal.
   * Should be used for asymmetric proposals.
   *
   * @return True if proposal is accepted, false otherwise.
   */
  template<class F>
  bool metropolisHastings(F* filter);

  /**
   * Apply Metropolis criterion to accept or reject proposal. May be used
   * in place of metropolisHastings() for symmetric proposals.
   *
   * @return True if proposal is accepted, false otherwise.
   */
  template<class F>
  bool metropolis(F* filter);

  /**
   * Accept proposal.
   *
   * @tparam F #concept::Filter type.
   *
   * @param filter Filter.
   */
  template<class F>
  void accept(F* filter);

  /**
   * Reject proposal.
   */
  void reject();

  /**
   * Adapt proposal.
   *
   * @tparam Q1 Conditional pdf type.
   *
   * @param[in,out] q Proposal to adapt.
   * @param sd Proposal adaptation scaling parameter. Zero gives default.
   * @param A Number of steps after which to start adaptation. Negative gives
   * no adaptation.
   *
   * Does nothing if the number of steps taken is less than @p A.
   */
  template<class Q1>
  void adapt(Q1& q, const real sd = 0.0, const int A = 0.0);

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
   *
   * @tparam L Location.
   *
   * @param theta Static state.
   */
  template<Location L>
  void term(Static<L>& theta);
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
   * Model.
   */
  B& m;

  /**
   * Random number generator.
   */
  Random& rng;

  /**
   * Timer.
   */
  TicToc timer;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Size of MCMC state. Will include at least all p-nodes, and potentially
   * the initial state of d- and c-nodes, depending on settings.
   */
  int M;

  /**
   * Is out not null?
   */
  bool haveOut;

  /**
   * Are initial conditions shared?
   */
  bool initialConditioned;

  /**
   * Prior distribution.
   */
  prior_type p0;

  /**
   * Current state.
   */
  ParticleMCMCState x1;

  /**
   * Previous or proposed state.
   */
  ParticleMCMCState x2;

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

  /**
   * Members required for proposal adaptation.
   */
  host_vector<> mu, sumMu;
  host_matrix<> Sigma, sumSigma;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
};

/**
 * Factory for creating ParticleMCMC objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ParticleMCMC
 */
template<Location CL = ON_HOST>
struct ParticleMCMCFactory {
  /**
   * Create particle MCMC sampler.
   *
   * @return ParticleMCMC object. Caller has ownership.
   *
   * @see ParticleMCMC::ParticleMCMC()
   */
  template<class B, class IO1>
  static ParticleMCMC<B,IO1,CL>* create(B& m, Random& rng,
      IO1* out = NULL, const InitialConditionType flag = INITIAL_SAMPLED) {
    return new ParticleMCMC<B,IO1,CL>(m, rng, out, flag);
  }
};
}

#include "Resampler.hpp"
#include "../math/misc.hpp"
#include "../misc/TicToc.hpp"

template<class B, class IO1, bi::Location CL>
bi::ParticleMCMC<B,IO1,CL>::ParticleMCMC(B& m, Random& rng, IO1* out,
    const InitialConditionType flag) :
    m(m),
    rng(rng),
    out(out),
    M(((flag == INITIAL_CONDITIONED) ? m.getNetSize(D_NODE) + m.getNetSize(C_NODE) : 0) + m.getNetSize(P_NODE)),
    haveOut(out != NULL),
    initialConditioned(flag == INITIAL_CONDITIONED),
    x1(m, M, out->size2()),
    x2(m, M, out->size2()),
    lastAccepted(false),
    accepted(0),
    total(0),
    mu(M),
    sumMu(M),
    Sigma(M,M),
    sumSigma(M,M) {
  /* construct prior */
  if (initialConditioned) {
    p0.set(0, m.getPrior(D_NODE));
    p0.set(1, m.getPrior(C_NODE));
    p0.set(2, m.getPrior(P_NODE));
  } else {
    p0.set(0, m.getPrior(P_NODE));
    // other components of p0 not set, but as their size is then zero, this is ok
  }
}

template<class B, class IO1, bi::Location CL>
IO1* bi::ParticleMCMC<B,IO1,CL>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL>
bi::ParticleMCMCState& bi::ParticleMCMC<B,IO1,CL>::getState() {
  return x1;
}

template<class B, class IO1, bi::Location CL>
typename bi::ParticleMCMC<B,IO1,CL>::prior_type& bi::ParticleMCMC<B,IO1,CL>::getPrior() {
  return p0;
}

template<class B, class IO1, bi::Location CL>
bi::ParticleMCMCState& bi::ParticleMCMC<B,IO1,CL>::getOtherState() {
  return x2;
}

template<class B, class IO1, bi::Location CL>
template<class Q1, class V1, bi::Location L, class F, class R>
void bi::ParticleMCMC<B,IO1,CL>::sample(Q1& q, const V1 x, const int C,
    const real T, Static<L>& theta, State<L>& s, F* filter, R* resampler,
    const real sd, const int A) {
  /* pre-conditions */
  assert (q.size() == M);
  assert (x.size() == M);

  int c;

  timer.tic();
  init(x, T, theta, s, filter, resampler);
  for (c = 0; c < C; ++c) {
    report(c);
    propose(q);
    prior();
    likelihood(T, theta, s, filter, resampler);
    metropolisHastings();
    adapt(q, sd, A);
    output(c);
  }
  term(theta);
}

template<class B, class IO1, bi::Location CL>
template<class V1, bi::Location L, class F, class R>
void bi::ParticleMCMC<B,IO1,CL>::init(const V1 x, const real T,
    Static<L>& theta, State<L>& s, F* filter, R* resampler) {
  /* pre-condition */
  assert (x.size() == M);

  proposal(x);
  prior();
  likelihood(T, theta, s, filter, resampler);
  accept(filter);
}

template<class B, class IO1, bi::Location CL>
template<class Q1>
void bi::ParticleMCMC<B,IO1,CL>::propose(Q1& q) {
  /* pre-condition */
  assert (q.size() == M);

  q.sample(rng, this->x1.theta, this->x2.theta);
  x2.lq = q.logDensity(this->x1.theta, this->x2.theta);
  x1.lq = q.logDensity(this->x2.theta, this->x1.theta);
}

template<class B, class IO1, bi::Location CL>
template<class V1>
void bi::ParticleMCMC<B,IO1,CL>::proposal(const V1 x) {
  /* pre-condition */
  assert (x.size() == M);

  this->x2.theta = x;
  x2.lq = 0.0;
  x1.lq = 0.0;
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMCMC<B,IO1,CL>::prior() {
  x2.lp = p0.logDensity(this->x2.theta);
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class R>
void bi::ParticleMCMC<B,IO1,CL>::likelihood(const real T,
    Static<L>& theta, State<L>& s, F* filter, R* resampler,
    const FilterType type) {
  /* pre-conditions */
  BI_ASSERT(filter->getOutput() != NULL,
      "Output required for filter used with ParticleMCMC");
  assert(out == NULL || filter->getOutput()->size2() == out->size2());

  if (initialConditioned) {
    set_rows(s.get(D_NODE), subrange(x2.theta, 0, ND));
    set_rows(s.get(C_NODE), subrange(x2.theta, ND, NC));
    row(theta.get(P_NODE), 0) = subrange(x2.theta, ND + NC, NP);
  } else {
    m.getPrior(D_NODE).samples(rng, s.get(D_NODE));
    m.getPrior(C_NODE).samples(rng, s.get(C_NODE));
    row(theta.get(P_NODE), 0) = x2.theta;
  }

//  BOOST_AUTO(x, host_map_vector(row(theta.get(P_NODE), 0)));
//  synchronize();
//  for (int i = 0; i < x->size(); ++i) {
//    std::cerr << (*x)(i) << ' ';
//  }
//  std::cerr << std::endl;
//  delete x;

  filter->reset();
  switch (type) {
    case UNCONDITIONED:
      filter->filter(T, theta, s, resampler);
      break;
    case CONDITIONED:
      filter->filter(T, theta, s, x1.xd, x1.xc, x1.xr, resampler);
      break;
  }
  x2.xs = row(theta.get(S_NODE), 0);
  filter->summarise(&x2.ll, &x2.lls, &x2.ess);

  /* summary */
  synchronize();
  x2.timeStamp = timer.toc();
}

template<class B, class IO1, bi::Location CL>
template<class F>
bool bi::ParticleMCMC<B,IO1,CL>::metropolisHastings(F* filter) {
  bool result;

  if (!IS_FINITE(x2.ll)) {
    result = false;
  } else if (!IS_FINITE(x1.ll)) {
    result = true;
  } else {
    real loglr = x2.ll - x1.ll;
    real logpr = x2.lp - x1.lp;
    real logqr = x1.lq - x2.lq;
    real logratio = loglr + logpr + logqr;

    result = log(rng.uniform<real>()) < logratio;
  }

  if (result) {
    accept(filter);
  } else {
    reject();
  }

  return result;
}

template<class B, class IO1, bi::Location CL>
template<class F>
bool bi::ParticleMCMC<B,IO1,CL>::metropolis(F* filter) {
  bool result;

  if (!IS_FINITE(x2.ll)) {
    result = false;
  } else if (!IS_FINITE(x1.ll)) {
    result = true;
  } else {
    real loglr = x2.ll - x1.ll;
    real logpr = x2.lp - x1.lp;
    real logratio = loglr + logpr;

    result = log(rng.uniform<real>()) < logratio;
  }

  if (result) {
    accept(filter);
  } else {
    reject();
  }

  return result;
}

template<class B, class IO1, bi::Location CL>
template<class F>
void bi::ParticleMCMC<B,IO1,CL>::accept(F* filter) {
  x1.swap(x2);
  ++accepted;
  ++total;
  lastAccepted = true;
  filter->sampleTrajectory(x2.xd, x2.xc, x2.xr);
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMCMC<B,IO1,CL>::reject() {
  ++total;
  lastAccepted = false;
}

template<class B, class IO1, bi::Location CL>
template<class Q1>
void bi::ParticleMCMC<B,IO1,CL>::adapt(Q1& q, const real sd,
    const int A) {
  real s;

  if (A >= 0) {
    BOOST_AUTO(theta, host_temp_vector<real>(m.getNetSize(P_NODE)));
    *theta = this->x1.theta;

    if (total == 1) {
      sumMu.clear();
      sumSigma.clear();
    }

    synchronize();
    logVec(*theta, q.getLogs());
    axpy(1.0, *theta, sumMu);
    syr(1.0, *theta, sumSigma);

    if (total > A) {
      if (sd <= 0.0) {
        s = std::pow(2.4,2) / m.getNetSize(P_NODE); // default
      } else {
        s = sd;
      }

      axpy(1.0 / total, sumMu, mu, true);
      Sigma = sumSigma;
      syr(-1.0*total, mu, Sigma);
      int j;
      for (j = 0; j < Sigma.size2(); ++j) {
        scal(s/(total - 1), column(Sigma, j));
      }
      q.setCov(Sigma);
    }
    delete theta;
  }
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMCMC<B,IO1,CL>::output(const int c) {
//  const int T = out->size2();
//  int n;
//  real t;

  if (haveOut) {
    out->writeSample(c, subrange(x1.theta, M - NP, NP)); // only p-node portion
    out->writeLogLikelihood(c, x1.ll);
    out->writeLogPrior(c, x1.lp);
    out->writeParticle(c, x1.xd, x1.xc, x1.xr);
    out->writeTimeLogLikelihoods(c, x1.lls);
    out->writeTimeEss(c, x1.ess);
    out->writeTimeStamp(c, x1.timeStamp);
//    if (c == 0) {
//      /* write time variable also */
//      for (n = 0; n < T; ++n) {
//        filter->getOutput()->readTime(n, t);
//        out->writeTime(n, t);
//      }
//    }
  }
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleMCMC<B,IO1,CL>::report(const int c) {
  std::cerr << c << ":\t";
  std::cerr.width(10);
  std::cerr << this->getState().ll;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getState().lp;
  std::cerr << "\tbeats\t";
  std::cerr.width(10);
  std::cerr << this->getOtherState().ll;
  std::cerr << '\t';
  std::cerr.width(10);
  std::cerr << this->getOtherState().lp;
  std::cerr << '\t';
  if (this->wasLastAccepted()) {
    std::cerr << "accept";
  }
  std::cerr << std::endl;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L>
void bi::ParticleMCMC<B,IO1,CL>::term(Static<L>& theta) {
  //
}

template<class B, class IO1, bi::Location CL>
inline bool bi::ParticleMCMC<B,IO1,CL>::wasLastAccepted() {
  return lastAccepted;
}

template<class B, class IO1, bi::Location CL>
inline int bi::ParticleMCMC<B,IO1,CL>::getNumSteps() {
  return total;
}

template<class B, class IO1, bi::Location CL>
inline int bi::ParticleMCMC<B,IO1,CL>::getNumAccepted() {
  return accepted;
}

#endif
