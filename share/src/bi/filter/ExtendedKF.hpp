/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_EXTENDEDKF_HPP
#define BI_FILTER_EXTENDEDKF_HPP

#include "../simulator/Simulator.hpp"
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
 * @tparam F Forcer type.
 * @tparam O Observer type.
 */
template<class B, class F, class O>
class ExtendedKF: public Simulator<B,F,O> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Forcer.
   * @param obs Observer.
   */
  ExtendedKF(B& m, F& in, O& obs);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::step()
   */
  template<class S1, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      S1& s, IO1& out) throw (CholeskyException);

  /**
   * @copydoc BootstrapPF::samplePath()
   */
  template<class S1, class IO1>
  void samplePath(Random& rng, S1& s, IO1& out);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Predict.
   *
   * @tparam S1 State type.
   *
   * @param rng Random number generator.
   * @param next Next step in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void predict(Random& rng, const ScheduleElement next, S1& s)
      throw (CholeskyException);

  /**
   * Correct prediction with observation to produce filter density.
   *
   * @tparam S1 State type.
   *
   * @param rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   *
   * @return Incremental log-likelihood.
   */
  template<class S1>
  void correct(Random& rng, const ScheduleElement now, S1& s)
      throw (CholeskyException);
  //@}

protected:
  /*
   * Sizes for convenience.
   */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NO = B::NO;
  static const int M = NR + ND;
};
}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/operation.hpp"
#include "../math/constant.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"

template<class B, class F, class O>
bi::ExtendedKF<B,F,O>::ExtendedKF(B& m, F& in, O& obs) :
    Simulator<B,F,O>(m, in, obs) {
  //
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::ExtendedKF<B,F,O>::samplePath(Random& rng, S1& s, IO1& out) {
  typedef typename loc_temp_vector<S1::location,real>::type vector_type;
  typedef typename loc_temp_matrix<S1::location,real>::type matrix_type;

  if (out.size() > 0) {
    subrange(s.times, 0, out.len) = out.timeCache.get(0, out.len);

    matrix_type U1(M, M), U2(M, M), C(M, M);
    vector_type mu1(M), mu2(M);

    int k = out.size();
    try {
      while (k > 0) {
        out.readCorrectedMean(k - 1, mu1);
        out.readCorrectedStd(k - 1, U1);

        if (k < out.size()) {
          out.readPredictedMean(k, mu2);
          out.readPredictedStd(k, U2);
          out.readCross(k, C);

          condition(mu1, U1, mu2, U2, C, column(s.path, k));
        }

        rng.gaussians(column(s.path, k - 1));
        trmv(U1, column(s.path, k - 1));
        axpy(1.0, mu1, column(s.path, k - 1));

        --k;
      }
    } catch (CholeskyException e) {
      BI_WARN_MSG(false,
          "Cholesky factorisation exception sampling trajectory");
    }
  }
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::ExtendedKF<B,F,O>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) throw (CholeskyException) {
  do {
    ++iter;
    this->predict(rng, *iter, s);
    this->correct(rng, *iter, s);
    this->output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());
}

template<class B, class F, class O>
template<class S1>
void bi::ExtendedKF<B,F,O>::predict(Random& rng, const ScheduleElement next,
    S1& s) throw (CholeskyException) {
  typedef typename loc_temp_matrix<S1::location,real>::type matrix_type;

  /* predict */
  Simulator<B,F,O>::predict(rng, next, s);

  /* predicted mean */
  s.mu1 = row(s.getDyn(), 0);

  /* across-time block of square-root covariance */
  columns(s.C, 0, NR).clear();
  subrange(s.C, 0, NR, NR, ND).clear();
  subrange(s.C, NR, ND, NR, ND) = subrange(s.F(), NR, ND, NR, ND);
  trmm(1.0, s.U2, s.C);

  /* current-time block of square-root covariance */
  rows(s.U1, NR, ND).clear();
  subrange(s.U1, 0, NR, 0, NR) = subrange(s.Q(), 0, NR, 0, NR);
  subrange(s.U1, 0, NR, NR, ND) = subrange(s.F(), 0, NR, NR, ND);
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
  ident(s.F());
  s.Q().clear();
}

template<class B, class F, class O>
template<class S1>
void bi::ExtendedKF<B,F,O>::correct(Random& rng, const ScheduleElement now,
    S1& s) throw (CholeskyException) {
  typedef typename loc_temp_matrix<S1::location,real>::type matrix_type;
  typedef typename loc_temp_vector<S1::location,real>::type vector_type;
  typedef typename loc_temp_vector<S1::location,int>::type int_vector_type;

  s.mu2 = s.mu1;
  s.U2 = s.U1;

  if (now.isObserved()) {
    BOOST_AUTO(mask, this->obs.getMask(now.indexObs()));
    const int W = mask.size();

    this->observe(rng, s);

    matrix_type C(M, W), U3(W, W), Sigma3(W, W), R3(W, W);
    vector_type y(W), z(W), mu3(W);
    int_vector_type map(W);

    /* construct projection from mask */
    Var* var;
    int id, start = 0, size;
    for (id = 0; id < this->m.getNumVars(O_VAR); ++id) {
      var = this->m.getVar(O_VAR, id);
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
    gather_columns(map, s.G(), C);
    gather_matrix(map, map, s.R(), R3);
    gather(map, row(s.get(O_VAR), 0), mu3);
    gather(map, row(s.get(OY_VAR), 0), y);

    trmm(1.0, s.U1, C);

    Sigma3.clear();
    syrk(1.0, C, 0.0, Sigma3, 'U', 'T');
    syrk(1.0, R3, 1.0, Sigma3, 'U', 'T');
    trmm(1.0, s.U1, C, 'L', 'U', 'T');
    chol(Sigma3, U3, 'U');

    /* update marginal log-likelihood */
    ///@todo Duplicates some operations in condition() calls below
    sub_elements(y, mu3, z);
    trsv(U3, z, 'U');
    s.logLikelihood += -0.5 * dot(z) - BI_HALF_LOG_TWO_PI
        - bi::log(prod_reduce(diagonal(U3)));

    if (now.indexTime() > 0) {
      condition(s.mu2, s.U2, mu3, U3, C, y);
    } else {
      condition(subrange(s.mu2, NR, ND), subrange(s.U2, NR, ND, NR, ND), mu3,
          U3, rows(C, NR, ND), y);
    }
    row(s.getDyn(), 0) = s.mu2;

    /* reset Jacobian */
    s.G().clear();
    s.R().clear();
  }
}

#endif
