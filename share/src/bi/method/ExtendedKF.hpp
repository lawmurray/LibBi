/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_EXTENDEDKF_HPP
#define BI_METHOD_EXTENDEDKF_HPP

#include "Simulator.hpp"
#include "Observer.hpp"
#include "misc.hpp"
#include "../state/ExtendedKFState.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Extended Kalman filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class IO1>
class ExtendedKF {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param out Output.
   */
  ExtendedKF(B& m, S* sim = NULL, IO1* out = NULL);

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
   * @copydoc BootstrapPF::filter(Random&, Schedule&, const V1, BootstrapPFState<B,L>&)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ExtendedKFState<B,L>& s, IO2* inInit)
          throw (CholeskyException);

  /**
   * @copydoc BootstrapPF::filter(Random&, Schedule&, const V1, BootstrapPFState<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, ExtendedKFState<B,L>& s)
          throw (CholeskyException);

  /**
   * @copydoc BootstrapPF::step(Random&, ScheduleIterator&, const ScheduleIterator, BootstrapPFState<B,L>&)
   */
  template<Location L>
  real step(ScheduleIterator& iter, const ScheduleIterator last,
      ExtendedKFState<B,L>& s) throw (CholeskyException);

  /**
   * @copydoc BootstrapPF::sampleTrajectory()
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
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(Random& rng, const ScheduleElement now, ExtendedKFState<B,L>& s,
      IO2* inInit);

  /**
   * Initialise, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param theta Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const ScheduleElement now, const V1 theta,
      ExtendedKFState<B,L>& s);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param next Next step in time schedule.
   * @param[in,out] s State.
   */
  template<Location L>
  void predict(const ScheduleElement next, ExtendedKFState<B,L>& s)
      throw (CholeskyException);

  /**
   * Correct prediction with observation to produce filter density.
   *
   * @tparam L Location.
   *
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<Location L>
  real correct(const ScheduleElement now, ExtendedKFState<B,L>& s)
      throw (CholeskyException);

  /**
   * Output static variables.
   *
   * @param L Location.
   *
   * @param s State.
   */
  template<Location L>
  void output0(const ExtendedKFState<B,L>& s);

  /**
   * Output.
   *
   * @tparam L Location.
   *
   * @param now Current step in time schedule.
   * @param s State.
   */
  template<Location L>
  void output(const ScheduleElement now, const ExtendedKFState<B,L>& s);

  /**
   * Output marginal log-likelihood estimate.
   *
   * @param ll Estimate of the marginal log-likelihood.
   */
  void outputT(const real ll);

  /**
   * Clean up.
   */
  void term();
  //@}

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

  /*
   * Sizes for convenience.
   */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NO = B::NO;
  static const int M = NR + ND;
};

/**
 * Factory for creating ExtendedKF objects.
 *
 * @ingroup method
 *
 * @see ExtendedKF
 */
struct ExtendedKFFactory {
  /**
   * Create extended Kalman filter.
   *
   * @return ExtendedKF object. Caller has ownership.
   *
   * @see ExtendedKF::ExtendedKF()
   */
  template<class B, class S, class IO1>
  static ExtendedKF<B,S,IO1>* create(B& m, S* sim = NULL, IO1* out = NULL) {
    return new ExtendedKF<B,S,IO1>(m, sim, out);
  }
};
}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/operation.hpp"
#include "../math/pi.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"

template<class B, class S, class IO1>
bi::ExtendedKF<B,S,IO1>::ExtendedKF(B& m, S* sim, IO1* out) :
    m(m), sim(sim), out(out) {
  //
}

template<class B, class S, class IO1>
inline S* bi::ExtendedKF<B,S,IO1>::getSim() {
  return sim;
}

template<class B, class S, class IO1>
inline void bi::ExtendedKF<B,S,IO1>::setSim(S* sim) {
  this->sim = sim;
}

template<class B, class S, class IO1>
inline IO1* bi::ExtendedKF<B,S,IO1>::getOutput() {
  return out;
}

template<class B, class S, class IO1>
inline void bi::ExtendedKF<B,S,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class S, class IO1>
template<bi::Location L, class IO2>
real bi::ExtendedKF<B,S,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ExtendedKFState<B,L>& s, IO2* inInit) throw (CholeskyException) {
  real ll = 0.0;

  ScheduleIterator iter = first;
  init(rng, *iter, s, inInit);
  output0(s);
  ll = correct(*iter, s);
  output(*iter, s);
  while (iter + 1 != last) {
    ll += step(iter, last, s);
  }
  term();
  outputT(ll);

  return ll;
}

template<class B, class S, class IO1>
template<bi::Location L, class V1>
real bi::ExtendedKF<B,S,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    ExtendedKFState<B,L>& s) throw (CholeskyException) {
  real ll = 0.0;

  ScheduleIterator iter = first;
  init(rng, *iter, theta, s);
  output0(s);
  ll = correct(*iter, s);
  output(*iter, s);
  while (iter + 1 != last) {
    ll += step(iter, last, s);
  }
  term();
  outputT(ll);

  return ll;
}

template<class B, class S, class IO1>
template<class M1>
void bi::ExtendedKF<B,S,IO1>::sampleTrajectory(Random& rng, M1 X) {
  typedef typename sim_temp_vector<M1>::type vector_type;
  typedef typename sim_temp_matrix<M1>::type matrix_type;

  matrix_type U1(M, M), U2(M, M), C(M, M);
  vector_type mu1(M), mu2(M);

  int k = out->size();
  try {
    while (k > 0) {
      out->readCorrectedMean(k - 1, mu1);
      out->readCorrectedStd(k - 1, U1);

      if (k < out->size()) {
        out->readPredictedMean(k, mu2);
        out->readPredictedStd(k, U2);
        out->readCross(k, C);

        condition(mu1, U1, mu2, U2, C, column(X, k));
      }

      rng.gaussians(column(X, k - 1));
      trmv(U1, column(X, k - 1));
      axpy(1.0, mu1, column(X, k - 1));

      --k;
    }
  } catch (CholeskyException e) {
    BI_WARN_MSG(false, "Cholesky factorisation exception sampling trajectory");
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class IO2>
void bi::ExtendedKF<B,S,IO1>::init(Random& rng, const ScheduleElement now,
    ExtendedKFState<B,L>& s, IO2* inInit) {
  ident(s.F);
  s.Q.clear();
  s.G.clear();
  s.R.clear();

  /* initialise */
  sim->init(rng, now, s, inInit);

  /* predicted mean */
  s.mu1 = row(s.getDyn(), 0);

  /* Cholesky factor of predicted covariance */
  s.U1 = s.Q;
  subrange(s.U1, 0, NR, NR, ND) = subrange(s.F, 0, NR, NR, ND);
  trmm(1.0, subrange(s.U1, 0, NR, 0, NR), subrange(s.U1, 0, NR, NR, ND));

  /* across-time covariance */
  s.C.clear();

  /* within-time covariance */
  if (out != NULL) {
    out->clear();
  }
}

template<class B, class S, class IO1>
template<bi::Location L, class V1>
void bi::ExtendedKF<B,S,IO1>::init(Random& rng, const ScheduleElement now,
    const V1 theta, ExtendedKFState<B,L>& s) {
  // this should be the same as init() above, but with a different call to
  // sim->init()
  ident(s.F);
  s.Q.clear();
  s.G.clear();
  s.R.clear();

  /* initialise */
  sim->init(rng, theta, now, s);

  /* predicted mean */
  s.mu1 = row(s.getDyn(), 0);

  /* Cholesky factor of predicted covariance */
  s.U1 = s.Q;
  subrange(s.U1, 0, NR, NR, ND) = subrange(s.F, 0, NR, NR, ND);
  trmm(1.0, subrange(s.U1, 0, NR, 0, NR), subrange(s.U1, 0, NR, NR, ND));

  /* across-time covariance */
  s.C.clear();

  /* within-time covariance */
  if (out != NULL) {
    out->clear();
  }
}

template<class B, class S, class IO1>
template<bi::Location L>
real bi::ExtendedKF<B,S,IO1>::step(ScheduleIterator& iter,
    const ScheduleIterator last, ExtendedKFState<B,L>& s)
        throw (CholeskyException) {
  do {
    ++iter;
    predict(*iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  real ll = correct(*iter, s);
  output(*iter, s);

  return ll;
}

template<class B, class S, class IO1>
template<bi::Location L>
void bi::ExtendedKF<B,S,IO1>::predict(const ScheduleElement next,
    ExtendedKFState<B,L>& s) throw (CholeskyException) {
  typedef typename loc_temp_matrix<L,real>::type matrix_type;

  /* predict */
  sim->advance(next, s);

  /* predicted mean */
  s.mu1 = row(s.getDyn(), 0);

  /* across-time block of square-root covariance */
  columns(s.C, 0, NR).clear();
  subrange(s.C, 0, NR, NR, ND).clear();
  subrange(s.C, NR, ND, NR, ND) = subrange(s.F, NR, ND, NR, ND);
  trmm(1.0, U2, s.C);

  /* current-time block of square-root covariance */
  rows(s.U1, NR, ND).clear();
  subrange(s.U1, 0, NR, 0, NR) = subrange(s.Q, 0, NR, 0, NR);
  subrange(s.U1, 0, NR, NR, ND) = subrange(s.F, 0, NR, NR, ND);
  trmm(1.0, subrange(s.U1, 0, NR, 0, NR), subrange(s.U1, 0, NR, NR, ND));

  /* predicted covariance */
  matrix_type Sigma(M, M);
  Sigma.clear();
  syrk(1.0, s.C, 0.0, Sigma, 'U', 'T');
  syrk(1.0, s.U1, 1.0, Sigma, 'U', 'T');

  /* across-time covariance */
  trmm(1.0, s.U2, s.C, 'L', 'U', 'T');

  /* Cholesky factor of predicted covariance */
  chol(Sigma, s.U1);

  /* reset Jacobian, as it has now been multiplied in */
  ident(s.F);
  s.Q.clear();
}

template<class B, class S, class IO1>
template<bi::Location L>
real bi::ExtendedKF<B,S,IO1>::correct(const ScheduleElement now,
    ExtendedKFState<B,L>& s) throw (CholeskyException) {
  typedef typename loc_temp_matrix<L,real>::type matrix_type;
  typedef typename loc_temp_vector<L,real>::type vector_type;
  typedef typename loc_temp_vector<L,int>::type int_vector_type;

  real ll = 0.0;
  s.mu2 = s.mu1;
  s.U2 = s.U1;

  if (now.isObserved()) {
    BOOST_AUTO(mask, sim->getObs()->getMask(now.indexObs()));
    const int W = mask.size();

    sim->observe(s);

    matrix_type C(M, W), U3(W, W), Sigma3(W, W), R3(W, W);
    vector_type y(W), z(W), mu3(W);
    int_vector_type map(W);

    /* construct projection from mask */
    Var* var;
    int id, start = 0, size;
    for (id = 0; id < m.getNumVars(O_VAR); ++id) {
      var = m.getVar(O_VAR, id);
      size = mask.getSize(id);

      if (mask.isSparse(id)) {
        addscal_elements(mask.getIndices(id), var->getStart(),
            subrange(map, start, size));
      } else {
        seq_elements(subrange(map, start, size), var->getStart());
      }
      start += size;
    }

    /* project matrices and vectors to active variables in mask */
    gather_columns(map, s.G, s.C);
    gather_matrix(map, map, s.R, R3);
    gather(map, row(s.get(O_VAR), 0), mu3);
    gather(map, row(s.get(OY_VAR), 0), y);

    trmm(1.0, s.1, s.C);

    Sigma3.clear();
    syrk(1.0, s.C, 0.0, Sigma3, 'U', 'T');
    syrk(1.0, R3, 1.0, Sigma3, 'U', 'T');
    trmm(1.0, s.U1, s.C, 'L', 'U', 'T');
    chol(Sigma3, U3, 'U');

    /* incremental log-likelihood */
    ///@todo Duplicates some operations in condition() calls below
    sub_elements(y, mu3, z);
    trsv(U3, z, 'U');
    ll = -0.5 * dot(z) - BI_HALF_LOG_TWO_PI
        - bi::log(prod_reduce(diagonal(U3)));

    if (now.indexTime() > 0) {
      condition(s.mu2, s.U2, mu3, U3, s.C, y);
    } else {
      condition(subrange(s.mu2, NR, ND), subrange(s.U2, NR, ND, NR, ND), mu3,
          U3, rows(s.C, NR, ND), y);
    }
    row(s.getDyn(), 0) = s.mu2;

    /* reset Jacobian */
    s.G.clear();
    s.R.clear();
  }

  return ll;
}

template<class B, class S, class IO1>
template<bi::Location L>
void bi::ExtendedKF<B,S,IO1>::output0(const ExtendedKFState<B,L>& s) {
  if (out != NULL) {
    out->writeParameters(s.get(P_VAR));
  }
}

template<class B, class S, class IO1>
template<bi::Location L>
void bi::ExtendedKF<B,S,IO1>::output(const ScheduleElement now,
    const ExtendedKFState<B,L>& s) {
  if (out != NULL && now.hasOutput()) {
    const int k = now.indexOutput();

    out->writeTime(k, now.getTime());
    out->writeState(k, s.getDyn());
    out->writePredictedMean(k, s.mu1);
    out->writePredictedStd(k, s.U1);
    out->writeCorrectedMean(k, s.mu2);
    out->writeCorrectedStd(k, s.U2);
    out->writeCross(k, s.C);
  }
}

template<class B, class S, class IO1>
void bi::ExtendedKF<B,S,IO1>::outputT(const real ll) {
  if (out != NULL) {
    out->writeLL(ll);
  }
}

template<class B, class S, class IO1>
void bi::ExtendedKF<B,S,IO1>::term() {
  sim->term();
}

#endif
