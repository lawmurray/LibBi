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
   * Filter forward, with fixed starting state.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param theta0 Parameters.
   * @param[out] s State.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class V1>
  real filter(Random& rng, const real T, const V1 theta0, State<B,L>& s)
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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO4>
  void init(Random& rng, State<B,L>& s, IO4* inInit);

  /**
   * Initialise, with fixed starting state.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param theta0 Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const V1 theta0, State<B,L>& s);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] s State.
   */
  template<Location L>
  void predict(const real tnxt, State<B,L>& s);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] s State.
   */
  template<Location L>
  void predictFixed(const real tnxt, State<B,L>& s);

  /**
   * Correct prediction with observation to produce filter density. Also
   * updated log-likelihood estimate.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void correct(State<B,L>& s) throw (CholeskyException);

  /**
   * Correct prediction from fixed state. As the covariance in such
   * situations is zero, this merely updates the log-likelihood estimate.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void correctFixed(State<B,L>& s) throw (CholeskyException);

  /**
   * Output.
   *
   * @tparam L Location.
   *
   * @param k Time index
   * @param s State.
   */
  template<Location L>
  void output(const int k, const State<B,L>& s);

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
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
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
    M(m.getNetSize(R_VAR) + m.getNetSize(D_VAR)) {
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
  assert (T >= state.t);

  int n = 0;
  state.ll = 0.0;
  init(rng, s, inInit);
  while (state.t < T) {
    predict(T, s);
    output(n, s);
    ++n;
    correct(s);
    output(n, s);
    ++n;
  }
  synchronize();
  term();

  return getLogLikelihood();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
real bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::filter(Random& rng,
    const real T, const V1 theta0, State<B,L>& s)
    throw (CholeskyException) {
  /* pre-conditions */
  assert (T >= state.t);

  int n = 0;
  state.ll = 0.0;

  init(rng, theta0, s);
  while (state.t < T) {
    if (theta0.size() == NP || state.t > 0) {
      predict(T, s);
    } else {
      predictFixed(T, s);
    }
    output(n, s);
    ++n;

    if (theta0.size() == NP || state.t > 0.0) {
      correct(s);
    } else {
      correctFixed(s);
    }
    output(n, s);
    ++n;
  }
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
template<bi::Location L, class IO4>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::init(Random& rng,
    State<B,L>& s, IO4* inInit) {
  sim.init(rng, s, inInit);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::init(Random& rng,
    const V1 theta0, State<B,L>& s) {
  sim.init(rng, theta0, s);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::predict(const real tnxt,
    State<B,L>& s) {
  real to = getNextObsTime(tnxt);
  const int P = s.size();

  /* zero appropriate blocks of square-root covariance */
  BOOST_AUTO(S, s.getStd());

  multi_syrk(P, 1.0, columns(S, NR, ND), 0.0, subrange(S, P*NR, P*ND, NR, ND), 'U', 'T');
  multi_potrf(P, subrange(S, P*NR, P*ND, NR, ND), 'U');
  rows(S, 0, P*NR).clear();

  /* simulate forward */
  while (state.t < to) {
    sim.advance(to, s);
    state.t = sim.getTime();
  }

  /* update observations */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
  }

  /* post-condition */
  assert (sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::predictFixed(const real tnxt,
    State<B,L>& s) {
  real to = getNextObsTime(tnxt);

  /* zero square-root covariance */
  BOOST_AUTO(S, s.getStd());
  S.clear();

  /* simulate forward */
  while (state.t < to) {
    sim.advance(to, s);
    state.t = sim.getTime();
  }

  /* post-condition */
  assert (sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::correct(State<B,L>& s)
    throw (CholeskyException) {
  /* update observations at current time */
  if (oyUpdater.getTime() == getTime()) {
    state.W = oyUpdater.getMask().size();

    //BOOST_AUTO(mask, oyUpdater.getMask());
    //BI_ERROR(W == mask.size() && W == observed.size(),
    //    "Previous prediction step does not match current correction step");

    BOOST_AUTO(S, s.getStd());

    typename loc_temp_matrix<L,real>::type Sigma(S.size1(), S.size2());
    typename loc_temp_vector<L,real>::type z(state.W);

    const int P = s.size();
    const int W = state.W;
    const int start = (getTime() > 0.0) ? 0 : m.getNetSize(R_VAR);
    const int size = (getTime() > 0.0) ? M : m.getNetSize(D_VAR);

    columns(S, M, W).clear();
    sim.observe(s);

    BOOST_AUTO(mu1, vec(columns(s.getDyn(), start, size)));
    BOOST_AUTO(mu2, vec(s.get(O_VAR)));
    BOOST_AUTO(U1, subrange(S, start, P*size, start, size));
    BOOST_AUTO(U2, subrange(S, P*M, P*W, M, W));
    BOOST_AUTO(S1, subrange(Sigma, start, P*size, start, size));
    BOOST_AUTO(S2, subrange(Sigma, P*M, P*W, M, W));
    BOOST_AUTO(C, subrange(Sigma, start, P*size, M, W));
    BOOST_AUTO(y, vec(s.get(OY_VAR)));

    Sigma.clear();
    multi_syrk(P, 1.0, S, 0.0, Sigma, 'U', 'T');
    multi_chol(P, S1, U1, 'U');
    multi_chol(P, S2, U2, 'U');

    z = y;
    axpy(-1.0, mu2, z);
    trsv(U2, z);
    state.ll += -0.5*dot(z) - BI_HALF_LOG_TWO_PI - BI_MATH_LOG(prod_reduce(diagonal(U2)));

    multi_condition(P, mu1, U1, mu2, U2, C, y);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::correctFixed(State<B,L>& s)
    throw (CholeskyException) {
  /* update observations at current time */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
    state.W = oyUpdater.getMask().size();

    //BOOST_AUTO(mask, oyUpdater.getMask());
    //BI_ERROR(W == mask.size() && W == observed.size(),
    //    "Previous prediction step does not match current correction step");

    const int P = s.size();
    const int W = state.W;

    BOOST_AUTO(S, s.getStd());
    typename loc_temp_matrix<L,real>::type U2(P*W, W);
    typename loc_temp_vector<L,real>::type z(state.W);

    S.clear();
    sim.observe(s);

    BOOST_AUTO(mu2, vec(s.get(O_VAR)));
    BOOST_AUTO(S2, subrange(S, P*M, P*W, M, W));
    BOOST_AUTO(y, vec(s.get(OY_VAR)));

    multi_syrk(P, 1.0, S2, 0.0, U2, 'U', 'T');
    multi_potrf(P, U2, 'U');

    z = y;
    axpy(-1.0, mu2, z);
    trsv(U2, z);
    state.ll += -0.5*dot(z) - BI_HALF_LOG_TWO_PI - BI_MATH_LOG(prod_reduce(diagonal(U2)));
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,IO1,IO2,IO3,CL>::output(const int k,
    const State<B,L>& s) {
  if (out != NULL) {
    sim.output(k, s);

    /* copy to contiguous buffer */
    typename temp_host_matrix<real>::type S(M, M);
    S = subrange(s.getStd(), 0, M, 0, M);
    synchronize(L == ON_DEVICE);
    out->writeStd(k, S);
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
