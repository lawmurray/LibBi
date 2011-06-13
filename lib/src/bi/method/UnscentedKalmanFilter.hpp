/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_UNSCENTEDKALMANFILTER_HPP
#define BI_METHOD_UNSCENTEDKALMANFILTER_HPP

#include "Simulator.hpp"
#include "misc.hpp"
#include "../pdf/ExpGaussianPdf.hpp"
#include "../updater/UnscentedRUpdater.hpp"
#include "../updater/UnscentedORUpdater.hpp"
#include "../updater/OUpdater.hpp"
#include "../updater/OYUpdater.hpp"
#include "../math/locatable.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * @internal
 *
 * State of UnscentedKalmanFilter.
 */
struct UnscentedKalmanFilterState {
  /**
   * Constructor.
   */
  UnscentedKalmanFilterState();

  /**
   * Current time.
   */
  real t;
};
}

bi::UnscentedKalmanFilterState::UnscentedKalmanFilterState() : t(0.0) {
  //
}

namespace bi {
/**
 * Unscented Kalman filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::UnscentedKalmanFilterBuffer type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL,
    StaticHandling SH>
class UnscentedKalmanFilter : public Markable<UnscentedKalmanFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param delta Time step for d- and r-nodes.
   * @param fUpdater Updater for f-net.
   * @param oyUpdater Updater for observations of o-net.
   */
  UnscentedKalmanFilter(B& m, const real delta = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~UnscentedKalmanFilter();

  /**
   * Get the current time.
   */
  real getTime();

  /**
   * @copydoc #concept::Filter::getOutput()
   */
  IO3* getOutput();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Filter forward.
   *
   * @tparam L Location.
   *
   * @param T Time to which to filter.
   * @param[out] theta Static state.
   * @param[out] s State.
   */
  template<Location L>
  void filter(const real T, Static<L>& theta, State<L>& s);

  /**
   * Filter forward with fixed starting state.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param T Time to which to filter.
   * @param x0 Starting state. Should contain d- and c-nodes, along with
   * p-nodes if <tt>SH == STATIC_OWN</tt>.
   * @param[out] theta Static state.
   * @param[out] s State.
   */
  template<Location L, class V1>
  void filter(const real T, const V1 x0, Static<L>& theta, State<L>& s);

  /**
   * @copydoc #concept::Filter::summarise()
   */
  template<class T1, class V2, class V3>
  void summarise(T1* ll, V2* lls, V3* ess);

  /**
   * @copydoc #concept::Filter::reset()
   */
  void reset();
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
   */
  template<Location L, class V1, class M1>
  void init(Static<L>& theta, ExpGaussianPdf<V1,M1>& corrected);

  /**
   * Forward prediction.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Time to which to predict.
   * @param corrected Corrected state marginal at current time.
   * @param[out] theta Static state.
   * @param[out] s State.
   * @param[out] observed Observation marginal at next time.
   * @param[out] uncorrected Uncorrected state marginal at next time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time when observations are available is reached.
   */
  template<Location L, class V1, class M1>
  void predict(const real tnxt, const ExpGaussianPdf<V1,M1>& corrected,
      Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
      ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY);

  /**
   * Forward prediction with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Time to which to predict.
   * @param x0 Starting state. Should contain d- and c-nodes, along with
   * p-nodes if <tt>SH == STATIC_OWN</tt>.
   * @param[out] theta Static state.
   * @param[out] s State.
   * @param[out] observed Observation marginal at next time.
   * @param[out] uncorrected Uncorrected state marginal at next time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time when observations are available is reached.
   */
  template<Location L, class V1, class M1, class V2>
  void predict(const real tnxt, const V2 x0,
      Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
      ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY);

  /**
   * Correct using observations at current time.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param uncorrected Uncorrected state marginal at current time.
   * @param SigmaXY Uncorrected-observed cross-covariance.
   * @param[out] State.
   * @param[out] observed Observation marginal at current time.
   * @param[out] corrected Corrected state marginal at current time.
   */
  template<Location L, class V1, class M1>
  void correct(const ExpGaussianPdf<V1,M1>& uncorrected, const M1& SigmaXY,
      State<L>& s, ExpGaussianPdf<V1,M1>& observed,
      ExpGaussianPdf<V1,M1>& corrected);

  /**
   * Output.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param uncorrected Uncorrected state marginal at current time.
   * @param corrected Corrected state marginal at current time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   */
  template<class V1, class M1>
  void output(const int k, const ExpGaussianPdf<V1,M1>& uncorrected,
      const ExpGaussianPdf<V1,M1>& corrected, M1& SigmaXX);

  /**
   * Clean up.
   *
   * @tparam L Location.
   */
  template<Location L>
  void term(Static<L>& theta);
  //@}

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

private:
  /**
   * Initialise observations.
   *
   * @tparam L Location.
   *
   * @param tnxt Time to which to predict.
   * @param s State.
   *
   * @return Time to which to predict.
   *
   * Sets #W.
   */
  template<Location L>
  real initObs(const real tnxt, State<L>& s);

  /**
   * Initialise unscented transformation.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param[out] theta Static state.
   * @param[out] s State.
   * @param[out] observed Observation marginal at next time.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   *
   * Sets #P, #lambda, #a, #Wm0, #Wc0, #Wi and resizes @p observed,
   * @p SigmaXY, @p s and @p theta as required.
   */
  template<Location L, class V1, class M1>
  void initTransform(Static<L>& theta, State<L>& s,
      ExpGaussianPdf<V1,M1>& observed, M1& SigmaXY);

  /**
   * Perform unscented transformation.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param[out] theta Static state.
   * @param[out] s State.
   * @param[out] observed Observation marginal at next time.
   * @param[out] uncorrected Uncorrected state marginal at next time.
   * @param[out] SigmaXX Uncorrected-corrected state cross-covariance.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   * @param tj Time to which to predict.
   * @param nsteps Number of random variate updates required.
   * @param mu0 Mean of corrected state at starting point.
   * @param X Sigma points of corrected state.
   */
  template<Location L, class V1, class M1, class V2, class M2>
  void transform(Static<L>& theta, State<L>& s,
      ExpGaussianPdf<V1,M1>& observed, ExpGaussianPdf<V1,M1>& uncorrected,
      M1 SigmaXX, M1 SigmaXY, const real tj, const int nsteps, const V2 mu0,
      M2 X);

  /**
   * Model.
   */
  B& m;

  /**
   * Time step.
   */
  real delta;

  /**
   * Size of state, including random variates and observations.
   */
  int N;

  /**
   * Number of observations at next time point.
   */
  int W;

  /**
   * Number of trajectories.
   */
  int P;

  /**
   * Unscented transformation parameters.
   */
  real alpha, beta, kappa, lambda, a;

  /**
   * Weights for unscented transformation.
   */
  real Wm0, Wc0, Wi;

  /**
   * Ids of updated o-nodes that are log-variables.
   */
  std::set<int> oLogs;

  /**
   * Updater for r-net.
   */
  UnscentedRUpdater<B,SH> rUpdater;

  /**
   * Updater for or-net.
   */
  UnscentedORUpdater<B,SH> orUpdater;

  /**
   * Updater for o-net.
   */
  OUpdater<B,SH> oUpdater;

  /**
   * Updater for oy-net.
   */
  OYUpdater<B,IO2,CL> oyUpdater;

  /**
   * Simulator.
   */
  Simulator<B,UnscentedRUpdater<B,SH>,IO1,SimulatorNetCDFBuffer,CL,SH> sim;

  /**
   * Output.
   */
  IO3* out;

  /**
   * State.
   */
  UnscentedKalmanFilterState state;

  /**
   * Estimate parameters as well as state?
   */
  bool haveParameters;

  /**
   * Is out not null?
   */
  bool haveOut;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NO = net_size<B,typename B::OTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
  static const int M = ND + NC + NR + ((SH == STATIC_OWN) ? NP : 0);
};

/**
 * Factory for creating UnscentedKalmanFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @see UnscentedKalmanFilter
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct UnscentedKalmanFilterFactory {
  /**
   * Create unscented Kalman filter.
   *
   * @return UnscentedKalmanFilter object. Caller has ownership.
   *
   * @see UnscentedKalmanFilter::UnscentedKalmanFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>* create(
      B& m, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>(m, delta, in, obs,
        out);
  }
};

}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/pi.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::UnscentedKalmanFilter(B& m,
    const real delta, IO1* in, IO2* obs, IO3* out) :
    m(m),
    delta(delta),
    N(0),
    W(0),
    P(0),
    alpha(1.0e-3),
    beta(2.0),
    kappa(0.0),
    oyUpdater(*obs),
    sim(m, delta, &rUpdater, in),
    out(out),
    haveParameters(SH == STATIC_OWN),
    haveOut(out != NULL) {
  reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::~UnscentedKalmanFilter() {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
inline real bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
inline IO3* bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s) {
  typedef typename locatable_vector<L,real>::type V1;
  typedef typename locatable_matrix<L,real>::type M1;

  ExpGaussianPdf<V1,M1> corrected(M);
  ExpGaussianPdf<V1,M1> uncorrected(M);
  ExpGaussianPdf<V1,M1> observed(0);
  M1 SigmaXX(M, M), SigmaXY(M, 0);

  int n = 0;

  init(theta, corrected);
  while (state.t < T) {
    predict(T, corrected, theta, s, observed, uncorrected, SigmaXX, SigmaXY);
    correct(uncorrected, SigmaXY, s, observed, corrected);
    output(n, uncorrected, corrected, SigmaXX);
    ++n;
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    const V1 x0, Static<L>& theta, State<L>& s) {
  typedef typename locatable_vector<L,real>::type V2;
  typedef typename locatable_matrix<L,real>::type M2;

  ExpGaussianPdf<V2,M2> corrected(M);
  ExpGaussianPdf<V2,M2> uncorrected(M);
  ExpGaussianPdf<V2,M2> observed(0);
  M2 SigmaXX(M, M), SigmaXY(M, 0);

  int n = 0;

  init(theta, corrected);
  while (state.t < T) {
    if (n == 0) {
      predict(T, x0, theta, s, observed, uncorrected, SigmaXX, SigmaXY);
    } else {
      predict(T, corrected, theta, s, observed, uncorrected, SigmaXX, SigmaXY);
    }
    correct(uncorrected, SigmaXY, s, observed, corrected);
    output(n, uncorrected, corrected, SigmaXX);
    ++n;
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class T1, class V2, class V3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::summarise(T1* ll,
    V2* lls, V3* ess) {
  /* pre-condition */
  BI_ERROR(out != NULL, "Cannot summarise UnscentedKalmanFilter without output");

  /* no ESS in this case */
  if (ess != NULL) {
    bi::fill(ess->begin(), ess->end(), 0.0);
  }

  if (lls != NULL || ll != NULL) {
    BOOST_AUTO(lls1, host_temp_vector<real>(out->size2()));
    out->readTimeLogLikelihoods(*lls1);

    /* log-likelihoods at each time */
    if (lls != NULL) {
      *lls = *lls1;
    }

    /* compute marginal log-likelihood */
    if (ll != NULL) {
      bi::sort(lls1->begin(), lls1->end());
      *ll = bi::sum(lls1->begin(), lls1->end(), 0.0);
    }

    delete lls1;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::init(Static<L>& theta,
    ExpGaussianPdf<V1,M1>& corrected) {
  sim.init(theta);

  /* restore prior mean over initial state */
  mean(m.template getPrior<D_NODE>(), subrange(corrected.mean(), 0, ND));
  mean(m.template getPrior<C_NODE>(), subrange(corrected.mean(), ND, NC));
  subrange(corrected.mean(), ND + NC, NR).clear();
  if (haveParameters) {
    mean(m.template getPrior<P_NODE>(), subrange(corrected.mean(), ND + NC + NR, NP));
  }

  /* restore prior covariance over initial state */
  corrected.cov().clear();
  cov(m.template getPrior<D_NODE>(), subrange(corrected.cov(), 0, ND, 0, ND));
  cov(m.template getPrior<C_NODE>(), subrange(corrected.cov(), ND, NC, ND, NC));
  ident(subrange(corrected.cov(), ND + NC, NR, ND + NC, NR));
  if (haveParameters) {
    cov(m.template getPrior<P_NODE>(), subrange(corrected.cov(), ND + NC + NR, NP, ND + NC + NR, NP));
  }

  corrected.init();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::predict(const real tnxt,
    const ExpGaussianPdf<V1,M1>& corrected,
    Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  /* pre-condition */
  assert (corrected.size() == M);

  /* initialise time interval and number of observations */
  real ti = state.t;
  real tj = initObs(tnxt, s);

  /* number of random variate updates required during time interval */
  int nsteps = num_steps(ti, tj, delta);

  /* required state size, state arranged in order: d-nodes, c-nodes,
   * r-nodes, p-nodes, o-nodes and finishing with extra r-nodes */
  N = M + W + NR*std::max(0, nsteps - 1);

  /* initialise unscented transformation */
  initTransform(theta, s, observed, SigmaXY);

  /* initialise state */
  BOOST_AUTO(X, temp_matrix<M1>(s.size(), M + W));
  set_rows(columns(*X, 0, M), corrected.mean());
  matrix_axpy(a, corrected.std(), subrange(*X, 1, M, 0, M));
  matrix_axpy(-a, corrected.std(), subrange(*X, N + 1, M, 0, M));
  ///@todo corrected.std() is upper triangular, needn't do full axpy.

  /* perform unscented transformation */
  transform(theta, s, observed, uncorrected, SigmaXX, SigmaXY, tj, nsteps, corrected.mean(), *X);

  delete X;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class V2>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::predict(const real tnxt,
    const V2 x0, Static<L>& theta, State<L>& s,
    ExpGaussianPdf<V1,M1>& observed, ExpGaussianPdf<V1,M1>& uncorrected,
    M1& SigmaXX, M1& SigmaXY) {
  /* pre-conditions */
  assert (x0.size() == M);

  /* initialise time interval and number of observations */
  real ti = state.t;
  real tj = initObs(tnxt, s);

  /* number of random variate updates required during time interval */
  int nsteps = num_steps(ti, tj, delta);

  /* required state size, state arranged in order: d-nodes, c-nodes,
   * r-nodes, p-nodes, o-nodes and finishing with extra r-nodes */
  N = W + NR*std::max(0, nsteps - 1);

  /* initialise unscented transformation */
  initTransform(theta, s, observed, SigmaXY);

  /* initialise state */
  BOOST_AUTO(X, temp_matrix<M1>(s.size(), M + W));
  set_rows(columns(*X, 0, M), x0);

  /* perform unscented transformation */
  transform(theta, s, observed, uncorrected, SigmaXX, SigmaXY, tj, nsteps, x0, *X);

  delete X;
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
real bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::initObs(const real tnxt,
    State<L>& s) {
  real to, ti = state.t;
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() >= ti) {
    oyUpdater.update(s);
    to = oyUpdater.getTime();
    W = oyUpdater.getMask().size();
  } else {
    to = tnxt;
    W = 0;
  }
  return to;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::initTransform(
    Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
    M1& SigmaXY) {
  /* number of sigma points */
  P = 2*N + 1;

  /* parameters */
  lambda = alpha*alpha*(N + kappa) - N;
  a = sqrt(N + lambda);

  /* weights */
  Wm0 = lambda/(N + lambda);
  Wc0 = Wm0 + (1.0 - alpha*alpha + beta);
  Wi = 0.5/(N + lambda);

  /* resize distributions */
  observed.resize(W, false);
  SigmaXY.resize(M, W, false);

  /* resize state (note may end up larger than P, see State) */
  s.resize(P);
  if (haveParameters) {
    theta.resize(P);
  } else {
    theta.resize(1, true);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class V2, class M2>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::transform(
    Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1 SigmaXX, M1 SigmaXY,
    const real tj, const int nsteps, const V2 mu0, M2 X) {
  typedef typename V1::value_type T1;
  /* pre-conditions */
  assert (mu0.size() == M);
  assert (X.size2() == M + W);
  assert (uncorrected.size() == M);
  assert (SigmaXX.size1() == M && SigmaXX.size2() == M);
  assert (SigmaXY.size1() == M);

  /* temporaries, column order for matrices is d-, c-, p-, o- then r-nodes */
  BOOST_AUTO(X1, &X);
  BOOST_AUTO(X2, temp_matrix<M1>(s.size(), M + W));
  BOOST_AUTO(Z1, rows(*X1, 0, P));
  BOOST_AUTO(Z2, rows(*X2, 0, P));
  BOOST_AUTO(mu, temp_vector<V1>(M + W));
  BOOST_AUTO(Sigma, temp_matrix<M1>(M + W, M + W));
  BOOST_AUTO(Wm, temp_vector<V1>(P));
  BOOST_AUTO(Wc, temp_vector<V1>(P));
  Sigma->clear();

  /**
   * Let \f$N\f$ be the dimensionality of the state and \f$V\f$ the
   * dimensionality of system noise (number of r-nodes). The state at time
   * \f$t_{n-1}\f$ is given by a Gaussian with mean \f$\boldsymbol{\mu}_x\f$
   * and covariance \f$\Sigma_x\f$. All matrices are stored column-major.
   *
   * Compute the weights:
   */
  *Wm->begin() = Wm0;
  *Wc->begin() = Wc0;
  bi::fill(Wm->begin() + 1, Wm->end(), Wi);
  bi::fill(Wc->begin() + 1, Wc->end(), Wi);

  /**
   * Compute the Cholesky decomposition of the covariance matrix,
   * \f$U \leftarrow \sqrt{\Sigma_x}\f$, where \f$U\f$ is an upper triangular
   * matrix.
   *
   * Set \f$\sigma\f$-points. For \f$i \leftarrow 0,\ldots,2N\f$:
   *
   * \f[\mathcal{X}_{n-1}^{(i)} \leftarrow \begin{cases}
   *   \boldsymbol{\mu}_x & i = 0 \\
   *   \boldsymbol{\mu}_x + a\,U^{(i,*)} & i = 1,\ldots,N \\
   *   \boldsymbol{\mu}_x - a\,U^{(i,*)} & i = N+1,\ldots,2N
   *   \end{cases}\,.
   * \f]
   *
   * where \f$U^{(i,*)}\f$ is the \f$i\f$th row of \f$U\f$, indexed from
   * zero, and \f$a = \sqrt{L + \lambda}\f$.
   *
   * Propagate system:
   *
   * \f[\mathcal{X}_n^{(i)} \leftarrow f(\mathcal{X}_{n-1}^{(i)})\,.\f]
   */
  s.get(D_NODE) = columns(*X1, 0, ND);
  exp_columns(s.get(D_NODE), m.getLogs(D_NODE));

  s.get(C_NODE) = columns(*X1, ND, NC);
  exp_columns(s.get(C_NODE), m.getLogs(C_NODE));

  s.get(R_NODE) = columns(*X1, ND + NC, NR);
  exp_columns(s.get(R_NODE), m.getLogs(R_NODE));

  if (haveParameters) {
    theta.get(P_NODE) = columns(*X1, ND + NC + NR, NP);
    exp_columns(theta.get(P_NODE), m.getLogs(P_NODE));
  }

  rUpdater.prepare(nsteps, N, a);
  if (haveParameters) {
    /* p-nodes changed, need to update s-nodes */
    sim.init(theta);
  }
  while (state.t < tj) {
    sim.advance(tj, s);
    state.t = sim.getTime();
  }

  /**
   * and observations:
   *
   * \f[\mathcal{Y}_n^{(i)} \leftarrow g(\mathcal{X}_n^{(i)})\f]
   */
  oLogs.clear();
  if (state.t >= tj) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    assert(W == mask.size());

    orUpdater.prepare(N, a);
    orUpdater.update(mask, s);
    oUpdater.update(mask, s);

    /* indices of observations at this time that are log-variables */
    int id, i;
    for (i = 0; i < W; ++i) {
      id = mask.id(i);
      if (m.isLog(O_NODE, id)) {
        oLogs.insert(id);
      }
    }
  }

  /* convert everything into log space */
  columns(*X2, 0, ND) = s.get(D_NODE);
  columns(*X2, ND, NC) = s.get(C_NODE);
  columns(*X2, ND + NC, NR) = s.get(R_NODE);
  if (haveParameters) {
    columns(*X2, ND + NC + NR, NP) = theta.get(P_NODE);
  }
  columns(*X2, M, W) = s.get(O_NODE);

  log_columns(columns(*X2, 0, ND), m.getLogs(D_NODE));
  log_columns(columns(*X2, ND, NC), m.getLogs(C_NODE));
  log_columns(columns(*X2, ND + NC, NR), m.getLogs(R_NODE));
  if (haveParameters) {
    log_columns(columns(*X2, ND + NC + NR, NP), m.getLogs(P_NODE));
  }
  log_columns(columns(*X2, M, W), oLogs);

  /**
   * Compute uncorrected mean:
   *
   * \f[\boldsymbol{\mu}_x \leftarrow \sum_i W_m^{(i)} \mathcal{X}_n^{(i)}\f]
   *
   * and observation mean:
   *
   * \f[\boldsymbol{\mu}_y \leftarrow \sum_i W_m^{(i)} \mathcal{Y}_n^{(i)}\f]
   */
  gemv(1.0, Z2, *Wm, 0.0, *mu, 'T');

  /**
   * Compute uncorrected covariance:
   *
   * \f[\Sigma_x \leftarrow \sum W_c^{(i)} (\mathcal{X}_n^{(i)} -
   * \boldsymbol{\mu}_x) (\mathcal{X}_n^{(i)} - \boldsymbol{\mu}_x)^T\,,\f]
   *
   * observation covariance:
   *
   * \f[\Sigma_y \leftarrow \sum W_c^{(i)} (\mathcal{Y}_n^{(i)} -
   * \boldsymbol{\mu}_y) (\mathcal{Y}_n^{(i)} - \boldsymbol{\mu}_y)^T\,,\f]
   *
   * and cross-covariance:
   *
   * \f[\Sigma_{yx} \leftarrow \sum W_c^{(i)} (\mathcal{Y}_n^{(i)} -
   * \boldsymbol{\mu}_y) (\mathcal{X}_n^{(i)} - \boldsymbol{\mu}_x)^T\f]
   */
  /* mean-adjust */
  sub_rows(columns(Z1, 0, M), mu0);
  sub_rows(Z2, *mu);

  /* uncorrected covariance */
  syrk(Wi, rows(Z2, 1, 2*N), 0.0, *Sigma, 'U', 'T');
  syr(Wc0, row(Z2, 0), *Sigma, 'U');

  /* corrected-uncorrected cross-covariance */
  gemm(Wi, subrange(Z1, 1, 2*N, 0, M), subrange(Z2, 1, 2*N, 0, M), 0.0, SigmaXX, 'T', 'N');
  ger(Wc0, subrange(row(Z1, 0), 0, M), subrange(row(Z2, 0), 0, M), SigmaXX);

  /* write results */
  uncorrected.mean() = subrange(*mu, 0, M);
  uncorrected.cov() = subrange(*Sigma, 0, M, 0, M);
  uncorrected.init();

  observed.mean() = subrange(*mu, M, W);
  observed.cov() = subrange(*Sigma, M, W, M, W);
  observed.setLogs(oLogs);
  observed.init();

  SigmaXY = subrange(*Sigma, 0, M, M, W);

  /* clean up */
  synchronize();
  delete X2;
  delete mu;
  delete Sigma;
  delete Wm;
  delete Wc;

  /* post-conditions */
  assert (sim.getTime() == state.t);
  assert (observed.size() == W);
  assert (SigmaXY.size1() == M && SigmaXY.size2() == W);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::correct(
    const ExpGaussianPdf<V1,M1>& uncorrected, const M1& SigmaXY, State<L>& s,
    ExpGaussianPdf<V1,M1>& observed, ExpGaussianPdf<V1,M1>& corrected) {
  if (W > 0 && oyUpdater.getTime() == state.t) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    BI_ERROR(W == mask.size() && W == observed.size(),
        "Previous prediction step does not match current correction step");

    /* condition state on observation */
    BOOST_AUTO(y, matrix_as_vector(s.get(OY_NODE)));
    condition(uncorrected, observed, SigmaXY, y, corrected);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::output(const int k,
    const ExpGaussianPdf<V1,M1>& uncorrected,
    const ExpGaussianPdf<V1,M1>& corrected, M1& SigmaXX) {
  if (haveOut) {
    out->writeTime(k, state.t);
    out->writeCorrectedState(k, corrected.mean(), corrected.cov());
    out->writeUncorrectedState(k, uncorrected.mean(), uncorrected.cov());
    out->writeCrossState(k, SigmaXX);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::term(Static<L>& theta) {
  sim.term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::mark() {
  Markable<UnscentedKalmanFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::restore() {
  Markable<UnscentedKalmanFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

#endif
