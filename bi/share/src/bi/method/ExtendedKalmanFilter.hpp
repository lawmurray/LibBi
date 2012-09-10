/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_EXTENDEDKALMANFILTER_HPP
#define BI_METHOD_EXTENDEDKALMANFILTER_HPP

#include "Simulator.hpp"
#include "misc.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"
#include "../updater/OYUpdater.hpp"

namespace bi {
/**
 * @internal
 *
 * State of ExtendedKalmanFilter.
 */
struct ExtendedKalmanFilterState {
  /**
   * Constructor.
   */
  ExtendedKalmanFilterState();

  /**
   * Current time.
   */
  real t;

  /**
   * Current log-likelihood.
   */
  real ll;

  /**
   * Observation size.
   */
  int W;
};
}

bi::ExtendedKalmanFilterState::ExtendedKalmanFilterState() :
    t(0.0), ll(0.0), W(0) {
  //
}

namespace bi {
/**
 * Extended Kalman filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::KalmanFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL>
class ExtendedKalmanFilter : public Markable<ExtendedKalmanFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param fUpdater Updater for f-net.
   * @param oyUpdater Updater for observations of o-net.
   */
  ExtendedKalmanFilter(B& m, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO4>
  real filter(Random& rng, const real T, State<B,L>& s, IO4* inInit)
      throw (CholeskyException);

  /**
   * Filter forward, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param theta Parameters.
   * @param[out] s State.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class V1>
  real filter(Random& rng, const real T, const V1 theta, State<B,L>& s)
      throw (CholeskyException);

  /**
   * @copydoc #concept::Filter::getLogLikelihood()
   */
  real getLogLikelihood();

  /**
   * @copydoc #concept::Filter::sampleTrajectory()
   */
  template<class M1>
  void sampleTrajectory(Random& rng, M1 xd, M1 xr);

  /**
   * @copydoc #concept::Filter::reset()
   */
  void reset();

  /**
   * @copydoc Simulator::getTime()
   */
  real getTime() const;

  /**
   * @copydoc Simulator::setDelta()
   */
  template<Location L>
  void setTime(const real t, State<B,L>& s);

  /**
   * @copydoc #concept::Filter::getOutput()
   */
  IO3* getOutput();

  /**
   * Get time of next observation.
   *
   * @param T Upper bound on time.
   */
  real getNextObsTime(const real T) const;
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
   * @tparam M1 Matrix type.
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param[out] S Square-root covariance matrix.
   * @param inInit Initialisation file.
   */
  template<Location L, class M1, class IO4>
  void init(Random& rng, State<B,L>& s, M1 U, M1 S, IO4* inInit);

  /**
   * Initialise, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param theta Parameters.
   * @param s State.
   * @param[out] S Square-root covariance matrix.
   */
  template<Location L, class V1, class M1>
  void init(Random& rng, const V1 theta, State<B,L>& s, M1 U, M1 S);

  /**
   * Predict.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] s State.
   * @param[in,out] S Square-root covariance matrix.
   */
  template<Location L, class M1>
  void predict(const real tnxt, State<B,L>& s, M1 U, M1 S);

  /**
   * Predict.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] s State.
   * @param[in,out] S Square-root covariance matrix.
   */
  template<Location L, class M1>
  void predictFixed(const real tnxt, State<B,L>& s, M1 U, M1 S);

  /**
   * Correct prediction with observation to produce filter density. Also
   * updated log-likelihood estimate.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param[in,out] s State.
   * @param[in,out] S Square-root covariance matrix.
   */
  template<Location L, class M1>
  void correct(State<B,L>& s, M1 U, M1 S) throw (CholeskyException);

  /**
   * Correct prediction from fixed state. As the covariance in such
   * situations is zero, this merely updates the log-likelihood estimate.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param[in,out] s State.
   * @param[in,out] S Square-root covariance matrix.
   */
  template<Location L, class M1>
  void correctFixed(State<B,L>& s, M1 U, M1 S) throw (CholeskyException);

  /**
   * Output.
   *
   * @tparam L Location.
   *
   * @param k Time index
   * @param s State.
   * @param S Square-root covariance matrix.
   */
  template<Location L, class M1>
  void output(const int k, const State<B,L>& s, const M1 U, M1 S);

  /**
   * Clean up.
   */
  void term();
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
   * Updater for oy-net.
   */
  OYUpdater<B,IO2,CL> oyUpdater;

  /**
   * Simulator.
   */
  Simulator<B,IO1,SimulatorNetCDFBuffer,CL> sim;

  /**
   * Output.
   */
  IO3* out;

  /**
   * State.
   */
  ExtendedKalmanFilterState state;

  /**
   * State size.
   */
  int M;

  /* net sizes, for convenience */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NP = B::NP;
  static const int NO = B::NO;
};

/**
 * Factory for creating ExtendedKalmanFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ExtendedKalmanFilter
 */
template<Location CL = ON_HOST>
struct ExtendedKalmanFilterFactory {
  /**
   * Create unscented Kalman filter.
   *
   * @return ExtendedKalmanFilter object. Caller has ownership.
   *
   * @see ExtendedKalmanFilter::ExtendedKalmanFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>* create(B& m, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL) {
    return new ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>(m, in, obs, out);
  }
};

}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/multi_operation.hpp"
#include "../math/pi.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::ExtendedKalmanFilter(B& m,
    IO1* in, IO2* obs, IO3* out) :
    m(m),
    oyUpdater(*obs),
    sim(m, in, out),
    out(out),
    M(NR + ND) {
  reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
inline real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::getTime() const {
  return state.t;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
inline void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::setTime(
    const real t, State<B,L>& s) {
  state.t = t;
  sim.setTime(t, s);
  oyUpdater.setTime(t, s);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
inline IO3* bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::getNextObsTime(const real T)
    const {
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() >= getTime() &&
      oyUpdater.getNextTime() < T) {
    return oyUpdater.getNextTime();
  } else {
    return T;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class IO4>
real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, State<B,L>& s, IO4* inInit) throw (CholeskyException) {
  /* pre-conditions */
  BI_ASSERT(T >= state.t);

  typedef typename loc_matrix<L,real>::type matrix_type;

  matrix_type U(s.size()*M, M), S(s.size()*M, M);
  int n = 0;
  state.ll = 0.0;

  init(rng, s, U, S, inInit);
  do {
    predict(T, s, U, S);
    output(n, s, U, S);
    ++n;
    correct(s, U, S);
    output(n, s, U, S);
    ++n;
  } while (state.t < T);
  synchronize();
  term();

  return getLogLikelihood();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, const V1 theta, State<B,L>& s)
    throw (CholeskyException) {
  /* pre-conditions */
  BI_ASSERT(T >= state.t);

  typedef typename loc_matrix<L,real>::type matrix_type;

  matrix_type U(s.size()*M, M), S(s.size()*M, M);
  int n = 0;
  state.ll = 0.0;

  init(rng, theta, s, U, S);
  do {
    predict(T, s, U, S);
    output(n, s, U, S);
    ++n;
    correct(s, U, S);
    output(n, s, U, S);
    ++n;
  } while (state.t < T);
  synchronize();
  term();

  return getLogLikelihood();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::sampleTrajectory(
    Random& rng, M1 xd, M1 xr) {
  ///@todo Implement.
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::reset() {
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1, class IO4>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::init(Random& rng,
    State<B,L>& s, M1 U, M1 S, IO4* inInit) {
  rows(s.getF(), 0, NR).clear();
  ident(rows(s.getF(), NR, ND));
  U.clear();
  S.clear();

  sim.init(rng, s, inInit);

  const int P = s.size();
  multi_gemm(P, 1.0, s.getQ(), s.getF(), 0.0, columns(S, NR, ND));
  multi_syrk(P, 1.0, columns(S, NR, ND), 0.0, subrange(U, P*NR, P*ND, NR, ND), 'U', 'T');
  multi_chol(P, subrange(U, P*NR, P*ND, NR, ND), subrange(U, P*NR, P*ND, NR, ND), 'U');
  s.getQ() = U;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::init(Random& rng,
    const V1 theta, State<B,L>& s, M1 U, M1 S) {
  rows(s.getF(), 0, NR).clear();
  ident(rows(s.getF(), NR, ND));
  U.clear();
  S.clear();

  sim.init(rng, theta, s);

  const int P = s.size();
  multi_gemm(P, 1.0, s.getQ(), s.getF(), 0.0, columns(S, NR, ND));
  multi_syrk(P, 1.0, columns(S, NR, ND), 0.0, subrange(U, P*NR, P*ND, NR, ND), 'U', 'T');
  multi_chol(P, subrange(U, P*NR, P*ND, NR, ND), subrange(U, P*NR, P*ND, NR, ND), 'U');
  s.getQ() = U;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::predict(const real tnxt,
    State<B,L>& s, M1 U, M1 S) {
  real to = getNextObsTime(tnxt);
  const int P = s.size();

  /* simulate forward */
  while (state.t < to) {
    /* clear Jacobian */
    rows(s.getF(), 0, NR).clear();
    ident(rows(s.getF(), NR, ND));

    /* drop noise terms from state square-root covariance */
    columns(S, NR, ND) = columns(U, NR, ND);
    multi_syrk(s.size(), 1.0, columns(S, NR, ND), 0.0, subrange(U, P*NR, P*ND, NR, ND), 'U', 'T');
    multi_chol(s.size(), subrange(U, P*NR, P*ND, NR, ND), subrange(U, P*NR, P*ND, NR, ND), 'U');
    rows(U, 0, P*NR).clear();
    s.getQ() = U;

    sim.advance(to, s);
    state.t = sim.getTime();

    subrange(S, 0, P*NR, 0, NR) = subrange(s.getQ(), 0, P*NR, 0, NR);
    columns(S, NR, ND) = s.getF();
    multi_trmm(s.size(), 1.0, s.getQ(), columns(S, NR, ND));
    multi_syrk(s.size(), 1.0, S, 0.0, U, 'U', 'T');
    if (state.t > 0) {
      multi_chol(s.size(), U, U, 'U');
    } else {
      multi_chol(s.size(), subrange(U, P*NR, P*ND, NR, ND), subrange(U, P*NR, P*ND, NR, ND), 'U');
    }
  }

  /* update observations */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
  }

  /* post-condition */
  BI_ASSERT(sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::predictFixed(const real tnxt,
    State<B,L>& s, M1 U, M1 S) {
  real to = getNextObsTime(tnxt);

  /* zero Jacobian */
  s.getJ().clear();

  /* simulate forward */
  do {
    sim.advance(to, s);
    state.t = sim.getTime();
  } while (state.t < to);

  /* post-condition */
  BI_ASSERT(sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::correct(State<B,L>& s,
    M1 U1, M1 S) throw (CholeskyException) {
  /* update observations at current time */
  if (oyUpdater.getTime() == getTime()) {
    state.W = oyUpdater.getMask().size();

    typename loc_temp_matrix<L,real>::type C(s.getG().size1(), s.getG().size2());
    typename loc_temp_matrix<L,real>::type U2(s.getR().size1(), s.getR().size2());
    typename loc_temp_vector<L,real>::type z(NO);

    const int P = s.size();
    const int W = state.W;

    s.getG().clear();
    sim.observe(s);

    C = s.getG();
    multi_trmm(P, 1.0, U1, C);

    U2.clear();
    multi_syrk(P, 1.0, C, 0.0, U2, 'U', 'T');
    multi_syrk(P, 1.0, s.getR(), 1.0, U2, 'U', 'T');
    multi_chol(P, U2, U2, 'U');
    multi_trmm(P, 1.0, U1, C, 'L', 'U', 'T'); // C now cross-covariance

    BOOST_AUTO(mu1, vec(s.getDyn()));
    BOOST_AUTO(mu2, vec(s.get(O_VAR)));
    BOOST_AUTO(y, vec(s.get(OY_VAR)));

    /* likelihood contribution */
    z = y;
    axpy(-1.0, mu2, z);
    trsv(U2, z);
    state.ll += -0.5*dot(z) - BI_HALF_LOG_TWO_PI - bi::log(prod_reduce(diagonal(U2)));

    if (state.t > 0) {
      multi_condition(P, mu1, U1, mu2, U2, C, y);
    } else {
      multi_condition(P, subrange(mu1, P*NR, P*ND), subrange(U1, P*NR, P*ND, NR, ND), mu2, U2, rows(C, P*NR, P*ND), y);
    }
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::correctFixed(State<B,L>& s,
    M1 U, M1 S) throw (CholeskyException) {
  /* update observations at current time */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
    state.W = oyUpdater.getMask().size();

    //BOOST_AUTO(mask, oyUpdater.getMask());
    //BI_ERROR(W == mask.size() && W == observed.size(),
    //    "Previous prediction step does not match current correction step");

    const int P = s.size();
    const int W = state.W;

    typename loc_temp_matrix<L,real>::type U2(P*W, W);
    typename loc_temp_vector<L,real>::type z(state.W);

    sim.observe(s);

    BOOST_AUTO(mu2, vec(s.get(O_VAR)));
    BOOST_AUTO(y, vec(s.get(OY_VAR)));

    multi_syrk(P, 1.0, s.getG(), 0.0, U2, 'U', 'T');
    multi_chol(P, U2, U2, 'U');

    z = y;
    axpy(-1.0, mu2, z);
    trsv(U2, z);
    state.ll += -0.5*dot(z) - BI_HALF_LOG_TWO_PI - bi::log(prod_reduce(diagonal(U2)));
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::output(const int k,
    const State<B,L>& s, const M1 U, M1 S) {
  if (out != NULL) {
    sim.output(k, s);

    /* copy to contiguous buffer */
    typename temp_host_matrix<real>::type U1(s.size()*M, M);
    U1 = s.getQ();
    synchronize(L == ON_DEVICE);
    out->writeStd(k, U1);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::getLogLikelihood() {
  return state.ll;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::term() {
  sim.term();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::mark() {
  Markable<ExtendedKalmanFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::restore() {
  Markable<ExtendedKalmanFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

#endif
