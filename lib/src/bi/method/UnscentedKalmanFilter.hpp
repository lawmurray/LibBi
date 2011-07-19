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
 * The state of UnscentedKalmanFilter includes all r-node updates, and thus
 * its %size changes according to the number of r-node updates required
 * across the time interval to the next observation. The leading components
 * of each Gaussian time-marginal provide, in order, the d-, c-, r- and
 * (optionally) p-nodes at the time of that marginal. Proceeding components
 * are the additional r-nodes, in chronological order, followed by predicted
 * observations.
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
   *
   * @param theta Static state.
   * @param[out] corrected Prior over initial state.
   */
  template<Location L, class V1, class M1>
  void init(Static<L>& theta, ExpGaussianPdf<V1,M1>& corrected);

  /**
   * Set up observations for upcoming transformation.
   *
   * @tparam L Location.
   *
   * @param T Upper bound on time for next observation.
   * @param[out] s State.
   *
   * @return Time of next observation, @p T if not further observations are
   * available within the time bound.
   */
  template<Location L>
  real obs(const real T, State<L>& s);

  /**
   * Set up parameters (sizes, weights, others) for upcoming transformation.
   *
   * @param tj End time for transformation.
   * @param fixed Is starting state fixed?
   *
   * Let \f$N\f$ be the dimensionality of the state and \f$V\f$ the
   * dimensionality of system noise (number of r-nodes). The state at time
   * \f$t_{n-1}\f$ is given by a Gaussian with mean \f$\boldsymbol{\mu}_x\f$
   * and covariance \f$\Sigma_x\f$. All matrices are stored column-major.
   *
   * Compute the weights:
   */
  void parameters(const real tj, const bool fixed = false);

  /**
   * Set up sigma points for upcoming transformation.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   *
   * @param corrected Corrected state marginal at current time.
   * @param[out] X1 Matrix of sigma points.
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
   */
  template<class V1, class M1, class M2>
  void sigmas(const ExpGaussianPdf<V1,M1>& corrected, M2 X1);

  /**
   * Set up sigma points for upcoming transformation with fixed starting
   * state.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param x0 Starting state.
   * @param[out] X1 Matrix of sigma points.
   */
  template<class V1, class M1>
  void sigmas(const V1 x0, M1 X1);

  /**
   * Resize required variables.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam M3 Matrix type.
   * @tparam M4 Matrix type.
   * @tparam V2 Vector type.
   * @tparam M5 Matrix type.
   *
   * @param[out] s State.
   * @param[out] theta Static state.
   * @param[out] observed Observation marginal.
   * @param[out] SigmaXY Uncorrected-observed cross-covariance.
   */
  template<Location L, class V1, class M1, class M2, class M3, class M4,
      class V2, class M5>
  void resize(State<L>& s, Static<L>& theta, ExpGaussianPdf<V1,M1>& observed,
      M2& SigmaXY, M3& X1, M4& X2, V2& mu, M5& Sigma);

  /**
   * Resize required variables for noise terms.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   * @tparam M3 Matrix type.
   *
   * @param[out] runcorrected Prediction density over noise terms.
   * @param[out] rcorrected Filter density over noise terms.
   * @param[out] SigmaRY Noise-observed cross-covariance.
   */
  template<class V1, class M1, class V2, class M2, class M3>
  void resizeNoise(ExpGaussianPdf<V1,M1>& runcorrected,
      ExpGaussianPdf<V2,M2>& rcorrected, M3& SigmaRY);

  /**
   * Propagate sigma points forward to next time.
   *
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam L Location.
   *
   * @param tj End time for transformation.
   * @param X1 Starting sigma points.
   * @param[out] Propagated sigma points.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   *
   * Propagate sigma points:
   *
   * \f[\mathcal{X}_n^{(i)} \leftarrow f(\mathcal{X}_{n-1}^{(i)})\,,\f]
   *
   * and observations:
   *
   * \f[\mathcal{Y}_n^{(i)} \leftarrow g(\mathcal{X}_n^{(i)})\,.\f]
   */
  template<class M1, class M2, Location L>
  void advance(const real tj, const M1 X1, M2 X2, Static<L>& theta,
      State<L>& s, const bool fixed = false);

  /**
   * Propagate sigma points forward to next time. This method saves some
   * copying to be slightly faster than its cousin when only noise terms
   * and observations will be of interest after the propagation.
   *
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam L Location.
   *
   * @param tj End time for transformation.
   * @param X1 Starting sigma points.
   * @param[out] Propagated sigma points.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   *
   * Propagate sigma points:
   *
   * \f[\mathcal{X}_n^{(i)} \leftarrow f(\mathcal{X}_{n-1}^{(i)})\,,\f]
   *
   * and observations:
   *
   * \f[\mathcal{Y}_n^{(i)} \leftarrow g(\mathcal{X}_n^{(i)})\,.\f]
   */
  template<class M1, class M2, Location L>
  void advanceNoise(const real tj, const M1 X1, M2 X2, Static<L>& theta,
      State<L>& s, const bool fixed = false);

  /**
   * Compute mean from propagated sigma points.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param X2 Propagated sigma points.
   * @param[out] mu2 Mean.
   *
   * Compute uncorrected mean:
   *
   * \f[\boldsymbol{\mu}_x \leftarrow \sum_i W_m^{(i)} \mathcal{X}_n^{(i)}\,,\f]
   *
   * and observation mean:
   *
   * \f[\boldsymbol{\mu}_y \leftarrow \sum_i W_m^{(i)} \mathcal{Y}_n^{(i)}\,.\f]
   */
  template<class M1, class V1>
  void mean(const M1 X2, V1 mu2);

  /**
   * Compute covariance from propagated sigma points.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   * @tparam M2 Matrix type.
   *
   * Compute uncorrected covariance:
   *
   * \f[\Sigma_x \leftarrow \sum W_c^{(i)} (\mathcal{X}_n^{(i)} -
   * \boldsymbol{\mu}_x) (\mathcal{X}_n^{(i)} - \boldsymbol{\mu}_x)^T\,,\f]
   *
   * and observation covariance:
   *
   * \f[\Sigma_y \leftarrow \sum W_c^{(i)} (\mathcal{Y}_n^{(i)} -
   * \boldsymbol{\mu}_y) (\mathcal{Y}_n^{(i)} - \boldsymbol{\mu}_y)^T\,.\f]
   */
  template<class M1, class V1, class M2>
  void cov(const M1 X2, const V1 mu2, M2 Sigma2);

  /**
   * Compute cross-covariance of starting and propagated sigma points.
   *
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam M3 Matrix type.
   *
   * @param X1 Starting sigma points.
   * @param X2 Propagated sigma points.
   * @param mu1 Starting state mean.
   * @param mu2 Propagated state mean.
   * @param[out] SigmaXX Cross-covariance.
   *
   * Compute cross-covariance:
   *
   * \f[\Sigma_{yx} \leftarrow \sum W_c^{(i)} (\mathcal{Y}_n^{(i)} -
   * \boldsymbol{\mu}_y) (\mathcal{X}_n^{(i)} - \boldsymbol{\mu}_x)^T.\f]
   */
  template<class M1, class M2, class V1, class V2, class M3>
  void cross(const M1 X1, const M2 X2, const V1 mu1, const V2 mu2,
      M3 SigmaXX);

  /**
   * Construct prediction density.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   * @tparam M3 Matrix type.
   *
   * @param mu Propagated state mean.
   * @param Sigma Propagated state covariance.
   * @param[out] uncorrected Prediction density.
   * @param[out] SigmaXY Prediction-observation cross-covariance.
   */
  template<class V1, class M1, class V2, class M2, class M3>
  void predict(const V1 mu, const M1 Sigma,
      ExpGaussianPdf<V2,M2>& uncorrected, M3& SigmaXY);

  /**
   * Construct prediction density over noise terms.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   * @tparam M3 Matrix type.
   *
   * @param mu Propagated state mean.
   * @param Sigma Propagated state covariance.
   * @param[out] uncorrected Prediction density over noise terms.
   * @param[out] SigmaXY Noise-observation cross-covariance.
   */
  template<class V1, class M1, class V2, class M2, class M3>
  void predictNoise(const V1 mu, const M1 Sigma,
      ExpGaussianPdf<V2,M2>& runcorrected, M3& SigmaRY);

  /**
   * Construct observation density.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param mu Propagated state mean.
   * @param Sigma Propagated state covariance.
   * @param[out] observed Observation density.
   */
  template<class V1, class M1, class V2, class M2>
  void observe(const V1 mu, const M1 Sigma, ExpGaussianPdf<V2,M2>& observed);

  /**
   * Correct prediction with observation to produce filter density.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam V3 Vector type.
   * @tparam M3 Matrix type.
   * @tparam V4 Vector type.
   * @tparam M4 Matrix type.
   *
   * @param uncorrected Prediction density.
   * @param SigmaXY Predicted-observed cross-covariance.
   * @param State.
   * @param observed Observation density.
   * @param[out] corrected Filter density.
   */
  template<Location L, class V1, class M1, class M2, class V3, class M3,
      class V4, class M4>
  void correct(const ExpGaussianPdf<V1,M1>& uncorrected, const M2& SigmaXY,
      State<L>& s, ExpGaussianPdf<V3,M3>& observed,
      ExpGaussianPdf<V4,M4>& corrected);

  /**
   * Correct noise prediction with observation to produce filter density over
   * noise terms.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam M2 Matrix type.
   * @tparam V3 Vector type.
   * @tparam M3 Matrix type.
   * @tparam V4 Vector type.
   * @tparam M4 Matrix type.
   *
   * @param runcorrected Prediction density over noise terms.
   * @param SigmaRY Noise-observed cross-covariance.
   * @param State.
   * @param observed Observation density.
   * @param[out] rcorrected Filter density over noise terms.
   */
  template<Location L, class V1, class M1, class M2, class V3, class M3,
      class V4, class M4>
  void correctNoise(const ExpGaussianPdf<V1,M1>& runcorrected,
      const M2& SigmaRY, State<L>& s, ExpGaussianPdf<V3,M3>& observed,
      ExpGaussianPdf<V4,M4>& rcorrected);

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
  template<class V1, class M1, class V2, class M2, class M3>
  void output(const int k, const ExpGaussianPdf<V1,M1>& uncorrected,
      const ExpGaussianPdf<V2,M2>& corrected, M3& SigmaXX);

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

protected:
  /**
   * Model.
   */
  B& m;

  /**
   * Total r-node updates required across next interval.
   */
  int nupdates;

  /**
   * Extra r-node updates required across next interval.
   */
  int nextra;

  /**
   * Size of unconditioned state.
   */
  int N1;

  /**
   * Size of full (conditioned and unconditioned) state.
   */
  int N2;

  /**
   * Number of observations.
   */
  int W;

  /**
   * Number of noise terms.
   */
  int V;

  /**
   * Number of sigma points.
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
   * Weight vectors.
   */
  host_vector<real> Wm, Wc;

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
   * Is out not null?
   */
  bool haveOut;

  /**
   * Estimate parameters as well as state?
   */
  static const bool haveParameters = SH == STATIC_OWN;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NO = net_size<B,typename B::OTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
  static const int M = ND + NC + NR + (haveParameters ? NP : 0);
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
    N1(0),
    N2(0),
    W(0),
    P(0),
    alpha(1.0e-3),
    beta(2.0),
    kappa(0.0),
    oyUpdater(*obs),
    sim(m, delta, &rUpdater, in),
    out(out),
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
  M1 X1, X2, Sigma;
  V1 mu;

  real tj;
  int n = 0;
  init(theta, corrected);
  while (state.t < T) {
    tj = obs(T, s);

    parameters(tj);
    resize(s, theta, observed, SigmaXY, X1, X2, mu, Sigma);
    sigmas(corrected, X1);
    advance(tj, X1, X2, theta, s);
    mean(X2, mu);
    cov(X2, mu, Sigma);
    if (haveOut) {
      cross(X1, X2, corrected.mean(), mu, SigmaXX);
    }
    predict(mu, Sigma, uncorrected, SigmaXY);
    observe(mu, Sigma, observed);
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
  M2 X1, X2, Sigma;
  V2 mu;

  real tj;
  int n = 0;
  init(theta, corrected);
  while (state.t < T) {
    tj = obs(T, s);

    if (n == 0) {
      parameters(tj, true);
    } else {
      parameters(tj, false);
    }
    resize(s, theta, observed, SigmaXY, X1, X2, mu, Sigma);
    if (n == 0) {
      sigmas(x0, X1);
      advance(tj, X1, X2, theta, s, true);
    } else {
      sigmas(corrected, X1);
      advance(tj, X1, X2, theta, s);
    }
    mean(X2, mu);
    cov(X2, mu, Sigma);
    if (haveOut) {
      if (n == 0) {
        cross(X1, X2, x0, mu, SigmaXX);
      } else {
        cross(X1, X2, corrected.mean(), mu, SigmaXX);
      }
    }
    predict(mu, Sigma, uncorrected, SigmaXY);
    observe(mu, Sigma, observed);
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
  bi::mean(m.template getPrior<D_NODE>(), subrange(corrected.mean(), 0, ND));
  bi::mean(m.template getPrior<C_NODE>(), subrange(corrected.mean(), ND, NC));
  subrange(corrected.mean(), ND + NC, NR).clear();
  if (haveParameters) {
    bi::mean(m.template getPrior<P_NODE>(), subrange(corrected.mean(), ND + NC + NR, NP));
  }

  /* restore prior covariance over initial state */
  corrected.cov().clear();
  bi::cov(m.template getPrior<D_NODE>(), subrange(corrected.cov(), 0, ND, 0, ND));
  bi::cov(m.template getPrior<C_NODE>(), subrange(corrected.cov(), ND, NC, ND, NC));
  ident(subrange(corrected.cov(), ND + NC, NR, ND + NC, NR));
  if (haveParameters) {
    bi::cov(m.template getPrior<P_NODE>(), subrange(corrected.cov(), ND + NC + NR, NP, ND + NC + NR, NP));
  }

  corrected.init();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
real bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::obs(const real T,
    State<L>& s) {
  real tj;
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() >= getTime() && oyUpdater.getNextTime() <= T) {
    oyUpdater.update(s);
    tj = oyUpdater.getTime();
    W = oyUpdater.getMask().size();
  } else {
    tj = T;
    W = 0;
  }
  return tj;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::parameters(
    const real tj, const bool fixed) {
  /* number of random variate updates required during time interval */
  real delta = sim.getDelta();
  nextra = std::max(0, lt_steps(tj, delta) - le_steps(state.t, delta));
  nupdates = lt_steps(tj, delta) - lt_steps(state.t, delta);

  /* required state size, state arranged in order: d-nodes, c-nodes,
   * r-nodes, p-nodes, intermediate r-nodes and o-nodes */
  if (fixed) {
    N1 = NR + NR*nextra + W;
    N2 = M + NR*nextra + W;
  } else {
    N1 = N2 = M + NR*nextra + W;
  }
  V = NR*nupdates;

  /* number of sigma points */
  P = 2*N1 + 1;

  /* parameters */
  lambda = alpha*alpha*(N1 + kappa) - N1;
  a = sqrt(N1 + lambda);

  /* weights */
  Wm0 = lambda/(N1 + lambda);
  Wc0 = Wm0 + (1.0 - alpha*alpha + beta);
  Wi = 0.5/(N1 + lambda);

  /* weight vectors */
  Wm.resize(P);
  Wc.resize(P);
  *Wm.begin() = Wm0;
  *Wc.begin() = Wc0;
  bi::fill(Wm.begin() + 1, Wm.end(), Wi);
  bi::fill(Wc.begin() + 1, Wc.end(), Wi);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class M2>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::sigmas(
    const ExpGaussianPdf<V1,M1>& corrected, M2 X1) {
  set_rows(columns(X1, 0, M), corrected.mean());
  if (nextra < nupdates) {
    /* current r-nodes remain */
    matrix_axpy(a, corrected.std(), subrange(X1, 1, M, 0, M));
    matrix_axpy(-a, corrected.std(), subrange(X1, 1 + N1, M, 0, M));
    ///@todo corrected.std() is upper triangular, needn't do full axpy.
  } else {
    /* new r-nodes */
    BOOST_AUTO(Sigma, temp_matrix<M1>(ND + NC, ND + NC));
    BOOST_AUTO(U, temp_matrix<M1>(ND + NC, ND + NC));
    *Sigma = subrange(corrected.cov(), 0, ND + NC, 0, ND + NC);
    chol(*Sigma, *U, 'U');
    matrix_axpy(a, *U, subrange(X1, 1, ND + NC, 0, ND + NC));
    matrix_axpy(-a, *U, subrange(X1, 1 + N1, ND + NC, 0, ND + NC));

    if (M1::on_device) {
      synchronize();
    }
    delete Sigma;
    delete U;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::sigmas(
    const V1 x0, M1 X1) {
  BOOST_AUTO(mu, duplicate_vector(x0));
  log_vector(subrange(*mu, 0, ND), m.getLogs(D_NODE));
  log_vector(subrange(*mu, ND, NC), m.getLogs(C_NODE));
  if (haveParameters) {
    log_vector(subrange(*mu, ND + NC + NR, NP), m.getLogs(P_NODE));
  }
  set_rows(columns(X1, 0, M), *mu);
  synchronize();
  delete mu;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class M2, class M3, class M4,
    class V2, class M5>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::resize(State<L>& s,
    Static<L>& theta, ExpGaussianPdf<V1,M1>& observed, M2& SigmaXY, M3& X1,
    M4& X2, V2& mu, M5& Sigma) {
  s.oresize(W, false);
  s.resize(P);
  if (haveParameters) {
    theta.resize(P);
  }
  observed.resize(W, false);
  SigmaXY.resize(N2 - W, W, false);
  X1.resize(s.size(), M, false);
  X2.resize(s.size(), N2, false);
  mu.resize(N2, false);
  Sigma.resize(N2, N2, false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class V2, class M2, class M3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::resizeNoise(
    ExpGaussianPdf<V1,M1>& runcorrected, ExpGaussianPdf<V2,M2>& rcorrected,
    M3& SigmaRY) {
  runcorrected.resize(V, false);
  rcorrected.resize(V, false);
  SigmaRY.resize(V, W, false);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class M2, bi::Location L>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::advance(const real tj,
    const M1 X1, M2 X2, Static<L>& theta, State<L>& s, const bool fixed) {
  /* copy to state */
  s.get(D_NODE) = columns(X1, 0, ND);
  s.get(C_NODE) = columns(X1, ND, NC);
  if (nextra == nupdates) {
    s.get(R_NODE) = columns(X1, ND + NC, NR);
  }
  if (haveParameters) {
    theta.get(P_NODE) = columns(X1, ND + NC + NR, NP);
  }

  /* exp relevant variables */
  exp_columns(s.get(D_NODE), m.getLogs(D_NODE));
  exp_columns(s.get(C_NODE), m.getLogs(C_NODE));
  if (haveParameters) {
    exp_columns(theta.get(P_NODE), m.getLogs(P_NODE));
  }

  /* propagate */
  rUpdater.prepare(nupdates, N1, W, a, fixed);
  if (haveParameters) {
    /* p-nodes changed, ensure s-nodes up-to-date */
    sim.init(theta);
  }
  int n = 0;
  while (state.t < tj) {
    sim.advance(std::min(gt_step(state.t, sim.getDelta()), tj), s);
    state.t = sim.getTime();

    if (n < nextra) {
      /* copy extra r-nodes now, others later */
      columns(X2, M + n*NR, NR) = s.get(R_NODE);
    }
    ++n;
  }

  /* observations */
  oLogs.clear();
  if (state.t >= tj && W > 0) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    assert(W == mask.size());

    orUpdater.prepare(N1, W, a, fixed);
    orUpdater.update(mask, s);
    oUpdater.update(mask, s);

    /* indices of observations at this time that are log-variables */
    int id, i;
    for (i = 0; i < W; ++i) {
      id = mask.id(i);
      if (m.isLog(O_NODE, id)) {
        oLogs.insert(i);
      }
    }
  }

  /* copy back to matrix */
  columns(X2, 0, ND) = s.get(D_NODE);
  columns(X2, ND, NC) = s.get(C_NODE);
  columns(X2, ND + NC, NR) = s.get(R_NODE);
  if (haveParameters) {
    columns(X2, ND + NC + NR, NP) = theta.get(P_NODE);
  }
  columns(X2, N2 - W, W) = s.get(O_NODE);

  /* log relevant variables */
  log_columns(columns(X2, 0, ND), m.getLogs(D_NODE));
  log_columns(columns(X2, ND, NC), m.getLogs(C_NODE));
  if (haveParameters) {
    log_columns(columns(X2, ND + NC + NR, NP), m.getLogs(P_NODE));
  }
  log_columns(columns(X2, N2 - W, W), oLogs);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class M2, bi::Location L>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::advanceNoise(
    const real tj, const M1 X1, M2 X2, Static<L>& theta, State<L>& s,
    const bool fixed) {
  /* copy to state */
  s.get(D_NODE) = columns(X1, 0, ND);
  s.get(C_NODE) = columns(X1, ND, NC);
  if (nextra == nupdates) {
    s.get(R_NODE) = columns(X1, ND + NC, NR);
  }
  if (haveParameters) {
    theta.get(P_NODE) = columns(X1, ND + NC + NR, NP);
  }

  /* exp relevant variables */
  exp_columns(s.get(D_NODE), m.getLogs(D_NODE));
  exp_columns(s.get(C_NODE), m.getLogs(C_NODE));
  if (haveParameters) {
    exp_columns(theta.get(P_NODE), m.getLogs(P_NODE));
  }

  /* propagate */
  rUpdater.prepare(nupdates, N1, W, a, fixed);
  if (haveParameters) {
    /* p-nodes changed, ensure s-nodes up-to-date */
    sim.init(theta);
  }
  int n = 0;
  while (state.t < tj) {
    sim.advance(std::min(gt_step(state.t, sim.getDelta()), tj), s);
    state.t = sim.getTime();

    if (n < nextra) {
      /* copy extra r-nodes now, others later */
      columns(X2, M + n*NR, NR) = s.get(R_NODE);
    }
    ++n;
  }

  /* observations */
  oLogs.clear();
  if (state.t >= tj && W > 0) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    assert(W == mask.size());

    orUpdater.prepare(N1, W, a, fixed);
    orUpdater.update(mask, s);
    oUpdater.update(mask, s);

    /* indices of observations at this time that are log-variables */
    int id, i;
    for (i = 0; i < W; ++i) {
      id = mask.id(i);
      if (m.isLog(O_NODE, id)) {
        oLogs.insert(i);
      }
    }
  }

  /* copy back to matrix */
  //columns(X2, 0, ND) = s.get(D_NODE);
  //columns(X2, ND, NC) = s.get(C_NODE);
  columns(X2, ND + NC, NR) = s.get(R_NODE);
  if (haveParameters) {
    columns(X2, ND + NC + NR, NP) = theta.get(P_NODE);
  }
  columns(X2, N2 - W, W) = s.get(O_NODE);

  /* log relevant variables */
  //log_columns(columns(X2, 0, ND), m.getLogs(D_NODE));
  //log_columns(columns(X2, ND, NC), m.getLogs(C_NODE));
  if (haveParameters) {
    log_columns(columns(X2, ND + NC + NR, NP), m.getLogs(P_NODE));
  }
  log_columns(columns(X2, N2 - W, W), oLogs);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class V1>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::mean(const M1 X2,
    V1 mu2) {
  /* pre-conditions */
  assert (X2.size2() == N2);
  assert (mu2.size() == N2);

  BOOST_AUTO(Z2, rows(X2, 0, P));
  gemv(1.0, Z2, Wm, 0.0, mu2, 'T');
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class V1, class M2>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::cov(
    const M1 X2, const V1 mu2, M2 Sigma2) {
  /* pre-conditions */
  assert (X2.size2() == N2);
  assert (mu2.size() == N2);
  assert (Sigma2.size1() == N2 && Sigma2.size2() == N2);

  BOOST_AUTO(Z2, rows(X2, 0, P));

  sub_rows(Z2, mu2);
  Sigma2.clear();
  syrk(Wi, rows(Z2, 1, 2*N1), 0.0, Sigma2, 'U', 'T');
  syr(Wc0, row(Z2, 0), Sigma2, 'U');
  add_rows(Z2, mu2); ///@todo Avoid.
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class M2, class V1, class V2, class M3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::cross(const M1 X1,
    const M2 X2, const V1 mu1, const V2 mu2, M3 SigmaXX) {
  /* pre-conditions */
  assert (X1.size2() == M);
  assert (X2.size2() == N2);
  assert (mu1.size() == M);
  assert (mu2.size() == N2);
  assert (SigmaXX.size1() == M && SigmaXX.size2() == M);

  BOOST_AUTO(Z1, rows(X1, 0, P));
  BOOST_AUTO(Z2, subrange(X2, 0, P, 0, M));

  sub_rows(Z1, mu1);
  sub_rows(Z2, subrange(mu2, 0, M));
  gemm(Wi, rows(Z1, 1, 2*N1), rows(Z2, 1, 2*N1), 0.0, SigmaXX, 'T', 'N');
  ger(Wc0, row(Z1, 0), row(Z2, 0), SigmaXX);
  //add_rows(Z1, mu1);
  //add_rows(Z2, subrange(mu2, 0, M));
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class V2, class M2, class M3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::predict(const V1 mu,
    const M1 Sigma, ExpGaussianPdf<V2,M2>& uncorrected, M3& SigmaXY) {
  /* pre-conditions */
  assert (Sigma.size1() == N2 && Sigma.size2() == N2);
  assert (uncorrected.size() == M);
  assert (SigmaXY.size1() == M && SigmaXY.size2() == W);

  uncorrected.mean() = subrange(mu, 0, M);
  uncorrected.cov() = subrange(Sigma, 0, M, 0, M);
  uncorrected.init();

  SigmaXY = subrange(Sigma, 0, M, N2 - W, W);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class V2, class M2, class M3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::predictNoise(const V1 mu,
    const M1 Sigma, ExpGaussianPdf<V2,M2>& runcorrected, M3& SigmaRY) {
  /* pre-conditions */
  assert (Sigma.size1() == N2 && Sigma.size2() == N2);
  assert (runcorrected.size() == V);
  assert (SigmaRY.size1() == V && SigmaRY.size2() == W);

  if (nupdates > 0) {
    subrange(runcorrected.mean(), 0, NR*nextra) = subrange(mu, M, NR*nextra);
    subrange(runcorrected.mean(), NR*nextra, NR) = subrange(mu, ND + NC, NR);

    subrange(runcorrected.cov(), 0, NR*nextra, 0, NR*nextra) = subrange(Sigma, M, NR*nextra, M, NR*nextra);
    subrange(runcorrected.cov(), NR*nextra, NR, NR*nextra, NR) = subrange(Sigma, ND + NC, NR, ND + NC, NR);
    transpose(subrange(Sigma, ND + NC, NR, M, NR*nextra), subrange(runcorrected.cov(), 0, NR*nextra, NR*nextra, NR));

    runcorrected.init();

    rows(SigmaRY, 0, NR*nextra) = subrange(Sigma, M, NR*nextra, N2 - W, W);
    rows(SigmaRY, NR*nextra, NR) = subrange(Sigma, ND + NC, NR, N2 - W, W);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class V2, class M2>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::observe(const V1 mu,
    const M1 Sigma, ExpGaussianPdf<V2,M2>& observed) {
  observed.mean() = subrange(mu, N2 - W, W);
  observed.cov() = subrange(Sigma, N2 - W, W, N2 - W, W);
  observed.setLogs(oLogs);
  observed.init();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class M2, class V3, class M3,
    class V4, class M4>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::correct(
    const ExpGaussianPdf<V1,M1>& uncorrected, const M2& SigmaXY, State<L>& s,
    ExpGaussianPdf<V3,M3>& observed, ExpGaussianPdf<V4,M4>& corrected) {
  /* pre-conditions */
  assert (uncorrected.size() == M);
  assert (SigmaXY.size1() == M && SigmaXY.size2() == W);
  assert (observed.size() == W);
  assert (corrected.size() == M);

  if (W > 0 && oyUpdater.getTime() == state.t) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    BI_ERROR(W == mask.size() && W == observed.size(),
        "Previous prediction step does not match current correction step");

    /* condition state on observation */
    condition(uncorrected, observed, SigmaXY, vec(s.get(OY_NODE)), corrected);
  } else {
    corrected = uncorrected;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class M1, class M2, class V3, class M3,
    class V4, class M4>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::correctNoise(
    const ExpGaussianPdf<V1,M1>& runcorrected, const M2& SigmaRY, State<L>& s,
    ExpGaussianPdf<V3,M3>& observed, ExpGaussianPdf<V4,M4>& rcorrected) {
  /* pre-conditions */
  assert (runcorrected.size() == V);
  assert (SigmaRY.size1() == V && SigmaRY.size2() == W);
  assert (observed.size() == W);
  assert (rcorrected.size() == V);

  if (W > 0 && oyUpdater.getTime() == state.t) {
    BOOST_AUTO(mask, oyUpdater.getMask());
    BI_ERROR(W == mask.size() && W == observed.size(),
        "Previous prediction step does not match current correction step");

    /* condition state on observation */
    condition(runcorrected, observed, SigmaRY, vec(s.get(OY_NODE)), rcorrected);
  } else {
    rcorrected = runcorrected;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL,
    bi::StaticHandling SH>
template<class V1, class M1, class V2, class M2, class M3>
void bi::UnscentedKalmanFilter<B,IO1,IO2,IO3,CL,SH>::output(const int k,
    const ExpGaussianPdf<V1,M1>& uncorrected,
    const ExpGaussianPdf<V2,M2>& corrected, M3& SigmaXX) {
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
