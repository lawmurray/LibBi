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
#include "Observer.hpp"
#include "misc.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * @internal
 *
 * State of ExtendedKalmanFilter.
 */
struct ExtendedKalmanFilterState {
  //
};
}

namespace bi {
/**
 * Extended Kalman filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class S, class IO1>
class ExtendedKalmanFilter: public Markable<ExtendedKalmanFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param out Output.
   */
  ExtendedKalmanFilter(B& m, S* sim = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get simulator.
   *
   * @return Simulator.
   */
  S* getSim();

  /**
   * Set simulator.
   *
   * @param sim Simulator.
   */
  void setSim(S* sim);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output.
   */
  void setOutput(IO1* out);

  /**
   * Filter forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param[out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO2>
  real filter(Random& rng, const real t, const real T, const int K,
      State<B,L>& s, IO2* inInit) throw (CholeskyException);

  /**
   * Filter forward, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param theta Parameters.
   * @param[out] s State.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class V1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s) throw (CholeskyException);

  /**
   * @copydoc #concept::Filter::sampleTrajectory()
   */
  template<class M1>
  void sampleTrajectory(Random& rng, M1 X);
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
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param s State.
   * @param[out] S Square-root covariance matrix.
   * @param inInit Initialisation file.
   */
  template<Location L, class M1, class IO2>
  void init(Random& rng, const real t, State<B,L>& s, M1 U, M1 S1,
      IO2* inInit);

  /**
   * Initialise, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param theta Parameters.
   * @param s State.
   * @param[out] S Square-root covariance matrix.
   */
  template<Location L, class V1, class M1>
  void init(Random& rng, const real t, const V1 theta, State<B,L>& s, M1 U,
      M1 S1);

  /**
   * Predict and correct.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param T Maximum time to which to advance.
   * @param[in,out] s State.
   * @param[in,out] U Cholesky factor of covariance matrix.
   * @param[in,out] S Square-root covariance matrix.
   *
   * @return Incremental log-likelihood.
   */
  template<Location L, class M1>
  real step(const real T, State<B,L>& s, M1 U, M1 S1);

  /**
   * Predict.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param T Maximum time to which to advance.
   * @param[in,out] s State.
   * @param[in,out] U Cholesky factor of covariance matrix.
   * @param[in,out] S Square-root covariance matrix.
   *
   * The filter is advanced to the soonest of @p T and the time of the next
   * observation.
   */
  template<Location L, class M1>
  void predict(const real T, State<B,L>& s, M1 U, M1 S1);

  /**
   * Correct prediction with observation to produce filter density. Also
   * updated log-likelihood estimate.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param[in,out] s State.
   * @param[in,out] S Square-root covariance matrix.
   *
   * @return Incremental log-likelihood.
   */
  template<Location L, class M1>
  real correct(State<B,L>& s, M1 U, M1 S1) throw (CholeskyException);

  /**
   * Output static variables.
   *
   * @param L Location.
   *
   * @param s State.
   */
  template<Location L>
  void output0(const State<B,L>& s);

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
  void output(const int k, const State<B,L>& s, const M1 U, M1 S1);

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

  /**
   * @copydoc concept::Markable::top()
   */
  void top();

  /**
   * @copydoc concept::Markable::pop()
   */
  void pop();

protected:
  /**
   * Model.
   */
  B& m;

  /**
   * Simulator.
   */
  S* sim;

  /**
   * Output.
   */
  IO1* out;

  /**
   * State.
   */
  ExtendedKalmanFilterState state;

  /*
   * Sizes for convenience.
   */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NO = B::NO;
};

/**
 * Factory for creating ExtendedKalmanFilter objects.
 *
 * @ingroup method
 *
 * @see ExtendedKalmanFilter
 */
struct ExtendedKalmanFilterFactory {
  /**
   * Create extended Kalman filter.
   *
   * @return ExtendedKalmanFilter object. Caller has ownership.
   *
   * @see ExtendedKalmanFilter::ExtendedKalmanFilter()
   */
  template<class B, class S, class IO1>
  static ExtendedKalmanFilter<B,S,IO1>* create(B& m, S* sim = NULL, IO1* out =
      NULL) {
    return new ExtendedKalmanFilter<B,S,IO1>(m, sim, out);
  }
};
}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/multi_operation.hpp"
#include "../math/pi.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"

template<class B, class S, class IO1>
bi::ExtendedKalmanFilter<B,S,IO1>::ExtendedKalmanFilter(B& m, S* sim,
    IO1* out) :
    m(m), sim(sim), out(out) {
  //
}

template<class B, class S, class IO1>
inline S* bi::ExtendedKalmanFilter<B,S,IO1>::getSim() {
  return sim;
}

template<class B, class S, class IO1>
inline void bi::ExtendedKalmanFilter<B,S,IO1>::setSim(S* sim) {
  this->sim = sim;
}

template<class B, class S, class IO1>
inline IO1* bi::ExtendedKalmanFilter<B,S,IO1>::getOutput() {
  return out;
}

template<class B, class S, class IO1>
inline void bi::ExtendedKalmanFilter<B,S,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class S, class IO1>
template<bi::Location L, class IO2>
real bi::ExtendedKalmanFilter<B,S,IO1>::filter(Random& rng, const real t,
    const real T, const int K, State<B,L>& s, IO2* inInit)
        throw (CholeskyException) {
  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  typedef typename loc_matrix<L,real>::type matrix_type;

  const int P = s.size();
  const int N = m.getDynSize();

  matrix_type U(P * N, N), S1(P * N, N);
  int k = 0, n = 0;
  real tk, ll = 0.0;

  init(rng, t, s, U, S1, inInit);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      ll += step(tk, s, U, S1);
      output(n++, s, U, S1);
    } while (sim->getTime() < tk);

    ++k;
  } while (k <= K);
  term();

  return ll;
}

template<class B, class S, class IO1>
template<bi::Location L, class V1>
real bi::ExtendedKalmanFilter<B,S,IO1>::filter(Random& rng, const real t,
    const real T, const int K, const V1 theta, State<B,L>& s)
        throw (CholeskyException) {
  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  typedef typename loc_matrix<L,real>::type matrix_type;

  const int P = s.size();
  const int N = m.getDynSize();

  matrix_type U(P * N, N), S1(P * N, N);
  int k = 0, n = 0;
  real tk, ll = 0.0;

  init(rng, t, theta, s, U, S1);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      ll += step(tk, s, U, S1);
      output(n++, s, U, S1);
    } while (sim->getTime() < tk);

    ++k;
  } while (k <= K);
  term();

  return ll;
}

template<class B, class S, class IO1>
template<class M1>
void bi::ExtendedKalmanFilter<B,S,IO1>::sampleTrajectory(Random& rng, M1 X) {
  ///@todo Implement.
}

template<class B, class S, class IO1>
template<bi::Location L, class M1, class IO2>
void bi::ExtendedKalmanFilter<B,S,IO1>::init(Random& rng, const real t,
    State<B,L>& s, M1 U, M1 S1, IO2* inInit) {
  rows(s.getF(), 0, NR).clear();
  ident(s.getF());
  U.clear();
  S1.clear();

  sim->init(rng, t, s, inInit);

  const int P = s.size();
  multi_gemm(P, 1.0, s.getQ(), columns(s.getF(), NR, ND), 0.0,
      columns(S1, NR, ND));
  multi_syrk(P, 1.0, columns(S1, NR, ND), 0.0,
      subrange(U, P * NR, P * ND, NR, ND), 'U', 'T');
  multi_chol(P, subrange(U, P * NR, P * ND, NR, ND),
      subrange(U, P * NR, P * ND, NR, ND), 'U');
  s.getQ() = U;

  if (out != NULL) {
    //out->clear();
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class V1, class M1>
void bi::ExtendedKalmanFilter<B,S,IO1>::init(Random& rng, const real t,
    const V1 theta, State<B,L>& s, M1 U, M1 S1) {
  rows(s.getF(), 0, NR).clear();
  ident(s.getF());
  U.clear();
  S1.clear();

  sim->init(rng, t, theta, s);

  const int P = s.size();
  multi_gemm(P, 1.0, s.getQ(), columns(s.getF(), NR, ND), 0.0,
      columns(S1, NR, ND));
  multi_syrk(P, 1.0, columns(S1, NR, ND), 0.0,
      subrange(U, P * NR, P * ND, NR, ND), 'U', 'T');
  multi_chol(P, subrange(U, P * NR, P * ND, NR, ND),
      subrange(U, P * NR, P * ND, NR, ND), 'U');
  s.getQ() = U;

  if (out != NULL) {
    //out->clear();
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class M1>
real bi::ExtendedKalmanFilter<B,S,IO1>::step(const real T, State<B,L>& s,
    M1 U, M1 S1) {
  predict(T, s, U, S1);
  return correct(s, U, S1);
}

template<class B, class S, class IO1>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,S,IO1>::predict(const real T, State<B,L>& s,
    M1 U, M1 S1) {
  const int P = s.size();

  /* clear Jacobian */
  rows(s.getF(), 0, NR).clear();
  ident(s.getF());

  /* drop noise terms from state square-root covariance */
  columns(S1, NR, ND) = columns(U, NR, ND);
  multi_syrk(s.size(), 1.0, columns(S1, NR, ND), 0.0,
      subrange(U, P * NR, P * ND, NR, ND), 'U', 'T');
  multi_chol(s.size(), subrange(U, P * NR, P * ND, NR, ND),
      subrange(U, P * NR, P * ND, NR, ND), 'U');
  rows(U, 0, P * NR).clear();
  s.getQ() = U;

  sim->advance(T, s);

  subrange(S1, 0, P * NR, 0, NR) = subrange(s.getQ(), 0, P * NR, 0, NR);
  columns(S1, NR, ND) = columns(s.getF(), NR, ND);
  multi_trmm(s.size(), 1.0, s.getQ(), columns(S1, NR, ND));
  multi_syrk(s.size(), 1.0, S1, 0.0, U, 'U', 'T');
  if (sim->getTime() > 0.0) {
    multi_chol(s.size(), U, U, 'U');
  } else {
    multi_chol(s.size(), subrange(U, P * NR, P * ND, NR, ND),
        subrange(U, P * NR, P * ND, NR, ND), 'U');
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class M1>
real bi::ExtendedKalmanFilter<B,S,IO1>::correct(State<B,L>& s, M1 U1, M1 S1)
    throw (CholeskyException) {
  real ll = 0.0;

  /* update observations at current time */
  if (sim->getObs() != NULL && sim->getObs()->isValid() && sim->getObs()->getTime() == sim->getTime()) {
    const int P = s.size();
    const int W = sim->getObs()->getMask().size();

    typename loc_temp_matrix<L,real>::type C(s.getG().size1(),
        s.getG().size2());
    typename loc_temp_matrix<L,real>::type U2(s.getR().size1(),
        s.getR().size2());
    typename loc_temp_vector<L,real>::type z(W);

    s.getG().clear();
    sim->observe(s);

    C = s.getG();
    multi_trmm(P, 1.0, U1, C);

    U2.clear();
    multi_syrk(P, 1.0, C, 0.0, U2, 'U', 'T');
    multi_syrk(P, 1.0, s.getR(), 1.0, U2, 'U', 'T');
    multi_chol(P, U2, U2, 'U');
    multi_trmm(P, 1.0, U1, C, 'L', 'U', 'T');  // C now cross-covariance

    BOOST_AUTO(mu1, vec(s.getDyn()));
    BOOST_AUTO(mu2, vec(s.get(O_VAR)));
    BOOST_AUTO(y, vec(s.get(OY_VAR)));

    /* likelihood contribution */
    z = y;
    axpy(-1.0, mu2, z);
    trsv(U2, z);
    ll = -0.5 * dot(z) - BI_HALF_LOG_TWO_PI
        - bi::log(prod_reduce(diagonal(U2)));

    if (sim->getTime() > 0.0) {
      multi_condition(P, mu1, U1, mu2, U2, C, y);
    } else {
      multi_condition(P, subrange(mu1, P * NR, P * ND),
          subrange(U1, P * NR, P * ND, NR, ND), mu2, U2,
          rows(C, P * NR, P * ND), y);
    }
  }

  return ll;
}

template<class B, class S, class IO1>
template<bi::Location L>
void bi::ExtendedKalmanFilter<B,S,IO1>::output0(const State<B,L>& s) {
  if (out != NULL) {
    out->writeParameters(s);
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class M1>
void bi::ExtendedKalmanFilter<B,S,IO1>::output(const int k,
    const State<B,L>& s, const M1 U, M1 S1) {
  if (out != NULL) {
    out->writeTime(k, sim->getTime());
    out->writeState(k, s);
    out->writeStd(k, s.getQ());
  }
}

template<class B, class S, class IO1>
void bi::ExtendedKalmanFilter<B,S,IO1>::term() {
  sim->term();
}

template<class B, class S, class IO1>
void bi::ExtendedKalmanFilter<B,S,IO1>::mark() {
  Markable<ExtendedKalmanFilterState>::mark(state);
  sim->mark();
}

template<class B, class S, class IO1>
void bi::ExtendedKalmanFilter<B,S,IO1>::restore() {
  Markable<ExtendedKalmanFilterState>::restore(state);
  sim->restore();
}

template<class B, class S, class IO1>
void bi::ExtendedKalmanFilter<B,S,IO1>::top() {
  Markable<ExtendedKalmanFilterState>::top(state);
  sim->top();
}

template<class B, class S, class IO1>
void bi::ExtendedKalmanFilter<B,S,IO1>::pop() {
  Markable<ExtendedKalmanFilterState>::pop(state);
  sim->pop();
}

#endif
