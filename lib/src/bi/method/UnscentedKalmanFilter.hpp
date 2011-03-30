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
template<class B, class IO1, class IO2, class IO3, Location CL, StaticHandling SH>
class UnscentedKalmanFilter : public Markable<UnscentedKalmanFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param fUpdater Updater for f-net.
   * @param oyUpdater Updater for observations of o-net.
   */
  UnscentedKalmanFilter(B& m, Random& rng, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL);

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
   * @copydoc #concept::Filter::filter()
   */
  template<Location L>
  void filter(const real T, Static<L>& theta, State<L>& s);

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
   * Model.
   */
  B& m;

  /**
   * Random number generator.
   */
  Random& rng;

  /**
   * Size of state, excluding random variates and observations.
   */
  int M;

  /**
   * Size of state, including random variates and observations.
   */
  int N;

  /**
   * Number of observations at next time point.
   */
  int W;

  /**
   * Unscented transformation parameters.
   */
  real alpha, beta, kappa, lambda, a;

  /**
   * Weights for unscented transformation given current #L.
   */
  real Wm0, Wc0, Wi;

  /**
   * Ids of updated o-nodes.
   */
  host_vector<int> ids;

  /**
   * Ids of updated o-nodes that are log-variables.
   */
  std::set<int> oLogs;

  /**
   * Updater for o-net.
   */
  OUpdater<B,SH> oUpdater;

  /**
   * Observation updater.
   */
  OYUpdater<B,IO2,CL> oyUpdater;

  /**
   * Simulator.
   */
  Simulator<B,IO1,SimulatorNetCDFBuffer,CL,SH> sim;

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
      B& m, Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, in, obs,
        out);
  }
};

}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/pi.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::UnscentedKalmanFilter(B& m,
    Random& rng, IO1* in, IO2* obs, IO3* out) :
    m(m),
    rng(rng),
    M(ND + NC + ((SH == STATIC_OWN) ? NP : 0)),
    N(0),
    W(0),
    alpha(1.0e-3),
    beta(2.0),
    kappa(0.0),
    oyUpdater(*obs),
    sim(m, NULL, in),
    out(out),
    haveParameters(SH == STATIC_OWN),
    haveOut(out != NULL) {
  reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::~UnscentedKalmanFilter() {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
inline real bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
inline IO3* bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
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
    correct(uncorrected, SigmaXY, s, observed, corrected);\
    output(n, uncorrected, corrected, SigmaXX);
    ++n;
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
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

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::init(Static<L>& theta,
    ExpGaussianPdf<V1,M1>& corrected) {
  sim.init(theta);

  /* restore prior mean over initial state */
  subrange(corrected.mean(), 0, ND) = m.getPrior(D_NODE).mean();
  subrange(corrected.mean(), ND, NC) = m.getPrior(C_NODE).mean();
  if (haveParameters) {
    subrange(corrected.mean(), ND + NC, NP) = m.getPrior(P_NODE).mean();
  }

  /* restore prior covariance over initial state */
  corrected.cov().clear();
  subrange(corrected.cov(), 0, ND, 0, ND) = m.getPrior(D_NODE).cov();
  subrange(corrected.cov(), ND, NC, ND, NC) = m.getPrior(C_NODE).cov();
  if (haveParameters) {
    subrange(corrected.cov(), ND + NC, NP, ND + NC, NP) = m.getPrior(P_NODE).cov();
  }

  corrected.init();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::predict(const real tnxt,
    const ExpGaussianPdf<V1,M1>& corrected,
    Static<L>& theta, State<L>& s, ExpGaussianPdf<V1,M1>& observed,
    ExpGaussianPdf<V1,M1>& uncorrected, M1& SigmaXX, M1& SigmaXY) {
  typedef typename V1::value_type T1;

  /* pre-conditions */
  assert (corrected.size() == M);
  assert (uncorrected.size() == M);
  assert (SigmaXX.size1() == M && SigmaXX.size2() == M);
  assert (SigmaXY.size1() == M);

  int i, nsteps, P;
  real to = (oyUpdater.getTime() >= state.t) ? oyUpdater.getTime() : tnxt;

  /* required size of state, depending on the number of discrete-time update
   * points over the time interval, and the number of observations at the
   * end */
  nsteps = floor(ceil(to) - state.t); // number random variate updates required (happen at integer times)
  W = oyUpdater.countCurrentNodes();
  N = M + W + NR*nsteps;
  P = 2*N + 1;

  if (L == ON_DEVICE) {
    /*
     * CUpdater on device requires that the number of trajectories
     * can be evenly split into thread blocks, which typically means a multiple
     * of 32 is needed. On device, then, we simulate additional mean
     * trajectories to pad up to a multiple of 32. This happens automatically
     * according to the size of the state, @c s, and has negligible performance
     * implications as we are simply filling empty capacity on the device. When
     * updating the mean and covariance of the state, however, we must be
     * careful to select only the meaningful subset of trajectories (i.e. the
     * first @p L, for unscented transformation operations.
     */
    P = ((P + 31)/32)*32;
  } else {
    /*
     * Likewise, when using SSE, the number of trajectories should be a
     * multiple of 4 (single precision) or 2 (double precision).
     */
    #ifdef USE_SSE
    P = ((P + BI_SSE_SIZE - 1)/BI_SSE_SIZE)*BI_SSE_SIZE;
    #endif
  }
  s.resize(P);
  if (haveParameters) {
    theta.resize(P);
  } else {
    theta.resize(1, true);
  }
  observed.resize(W, false);
  SigmaXY.resize(M, W, false);

  /* temporaries, column order for matrices is d-, c-, p-, o- then r-nodes */
  BOOST_AUTO(X1, temp_matrix<M1>(P, M + W));
  BOOST_AUTO(X2, temp_matrix<M1>(P, M + W));
  BOOST_AUTO(Z1, rows(*X1, 0, 2*N + 1));
  BOOST_AUTO(Z2, rows(*X2, 0, 2*N + 1));
  BOOST_AUTO(mu, temp_vector<V1>(M + W));
  BOOST_AUTO(Sigma, temp_matrix<M1>(M + W, M + W));
  BOOST_AUTO(Wm, temp_vector<V1>(2*N + 1));
  BOOST_AUTO(Wc, temp_vector<V1>(2*N + 1));
  Sigma->clear();

  /* zero random variates, subsequent operations only on leading diagonals */
  s.get(R_NODE).clear();
  s.get(OR_NODE).clear();

  /* unscented transform configuration */
  lambda = alpha*alpha*(N + kappa) - N;
  a = sqrt(N + lambda);

  Wm0 = lambda/(N + lambda);
  Wc0 = Wm0 + (1.0 - alpha*alpha + beta);
  Wi = 0.5/(N + lambda);

  *Wm->begin() = Wm0;
  *Wc->begin() = Wc0;
  bi::fill(Wm->begin() + 1, Wm->end(), Wi);
  bi::fill(Wc->begin() + 1, Wc->end(), Wi);

  /**
   * Let \f$N\f$ be the dimensionality of the state and \f$V\f$ the
   * dimensionality of system noise (number of r-nodes). The state at time
   * \f$t_{n-1}\f$ is given by a Gaussian with mean \f$\boldsymbol{\mu}_x\f$
   * and covariance \f$\Sigma_x\f$. All matrices are stored column-major.
   *
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
   * @todo corrected.std() is upper triangular, needn't do full axpy.
   */
  set_rows(columns(*X1, 0, M), corrected.mean());
  matrix_axpy(a, corrected.std(), subrange(*X1, 1, M, 0, M));
  matrix_axpy(-a, corrected.std(), subrange(*X1, N + 1, M, 0, M));

  /**
   * Propagate system:
   *
   * \f[\mathcal{X}_n^{(i)} \leftarrow f(\mathcal{X}_{n-1}^{(i)})\,.\f]
   *
   * and observations:
   *
   * \f[\mathcal{Y}_n^{(i)} \leftarrow g(\mathcal{X}_n^{(i)})\f]
   */
  s.get(D_NODE) = columns(*X1, 0, ND);
  expCols(s.get(D_NODE), m.getPrior(D_NODE).getLogs());

  s.get(C_NODE) = columns(*X1, ND, NC);
  expCols(s.get(C_NODE), m.getPrior(C_NODE).getLogs());

  if (haveParameters) {
    theta.get(P_NODE) = columns(*X1, ND + NC, NP);
    expCols(theta.get(P_NODE), m.getPrior(P_NODE).getLogs());
  }

  int step = 0, nr = (nsteps > 0) ? NR : 0;
  while (state.t < to) {
    /* generate random variates if need be */
    BOOST_AUTO(d1, diagonal(rows(s.get(R_NODE), 1 + M + W + step*nr, nr)));
    BOOST_AUTO(d2, diagonal(rows(s.get(R_NODE), 1 + N + M + W + step*nr, nr)));
    if (state.t == floor(state.t)) {
      bi::fill(d1.begin(), d1.end(), a);
      bi::fill(d2.begin(), d2.end(), -a);
    }

    /* simulate forward */
    if (haveParameters) {
      /* p-nodes changed, need to update s-nodes */
      sim.init(theta);
    }
    sim.advance(std::min(to, (real)floor(state.t + 1.0)), s);

    /* clear random variates again for next step if need be */
    if (state.t == floor(state.t)) {
      d1.clear();
      d2.clear();
      ++step;
    }

    state.t = sim.getTime();
  }
  assert (step == nsteps);

  oLogs.clear();
  if (oyUpdater.getTime() >= state.t) {
    oyUpdater.getCurrentNodes(ids);
    assert(W == ids.size());

    BOOST_AUTO(d1, diagonal(rows(s.get(OR_NODE), 1 + M, W)));
    BOOST_AUTO(d2, diagonal(rows(s.get(OR_NODE), 1 + N + M, W)));
    bi::fill(d1.begin(), d1.end(), a);
    bi::fill(d2.begin(), d2.end(), -a);

    oUpdater.update(ids, s);

    /* indices of observations at this time that are log-variables */
    for (i = 0; i < ids.size(); ++i) {
      if (m.getPrior(O_NODE).getLogs().find(ids[i]) != m.getPrior(O_NODE).getLogs().end()) {
        oLogs.insert(i);
      }
    }
  }

  logCols(s.get(D_NODE), m.getPrior(D_NODE).getLogs());
  logCols(s.get(C_NODE), m.getPrior(C_NODE).getLogs());
  logCols(s.get(O_NODE), oLogs);
  if (haveParameters) {
    logCols(theta.get(P_NODE), m.getPrior(P_NODE).getLogs());
  }

  columns(*X2, 0, ND) = s.get(D_NODE);
  columns(*X2, ND, NC) = s.get(C_NODE);
  if (haveParameters) {
    columns(*X2, ND + NC, NP) = theta.get(P_NODE);
  }
  columns(*X2, M, W) = columns(s.get(O_NODE), 0, W);

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
  sub_rows(columns(Z1, 0, M), corrected.mean());
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
  delete X1;
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

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::correct(
    const ExpGaussianPdf<V1,M1>& uncorrected, const M1& SigmaXY, State<L>& s,
    ExpGaussianPdf<V1,M1>& observed, ExpGaussianPdf<V1,M1>& corrected) {
  if (oyUpdater.getTime() == state.t) {
    assert (W > 0);
    BOOST_AUTO(y, temp_vector<V1>(W)); // vector for contiguous observations of selected o-nodes
    oyUpdater.update(s, *y);

    BI_ERROR(W == ids.size() && W == observed.size(),
        "Previous prediction step does not match current correction step");

    host_vector<real> diff(W);
    diff = *y;
    logVec(diff, oLogs);
    axpy(-1.0, observed.mean(), diff);

    /* condition state on observation */
    condition(uncorrected, observed, SigmaXY, *y, corrected);

    delete y;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
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

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::term(Static<L>& theta) {
  sim.term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::mark() {
  Markable<UnscentedKalmanFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::restore() {
  Markable<UnscentedKalmanFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

#endif
