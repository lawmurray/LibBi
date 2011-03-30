/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1246 $
 * $Date: 2011-01-31 16:23:46 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_METHOD_PIGGYBACKPARTICLEFILTER_HPP
#define BI_METHOD_PIGGYBACKPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "UnscentedKalmanFilter.hpp"
#include "../buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "../updater/OUpdater.hpp"
#include "../updater/ORUpdater.hpp"

namespace bi {
/**
 * PiggyBack particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::PiggyBackParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class PiggyBackParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  PiggyBackParticleFilter(B& m, Random& rng, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc #concept::Filter::filter()
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL);

  /**
   * @copydoc #concept::Filter::filter(real, const V1&, R*)
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Look ahead to next observation.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   *
   * Looks forward to the next observation, setting r-nodes in @p s to improve
   * support at this next time, and adjusting log-weights @p lws accordingly.
   */
  template<Location L, class V1, class V2, class M2>
  void lookahead(State<L>& s, V1& lws, ExpGaussianPdf<V2,M2>& corrected,
      ExpGaussianPdf<V2,M2>& observed, ExpGaussianPdf<V2,M2>& uncorrected,
      M2& SigmaXX, M2& SigmaXY1, M2& SigmaXY2);
  //@}

private:
  /**
   * Unscented Kalman filter for proposals.
   */
  UnscentedKalmanFilter<B,IO1,IO2,UnscentedKalmanFilterNetCDFBuffer,ON_HOST> ukf;
};

/**
 * Factory for creating PiggyBackParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see PiggyBackParticleFilter
 */
template<Location CL = ON_HOST>
struct PiggyBackParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return PiggyBackParticleFilter object. Caller has ownership.
   *
   * @see PiggyBackParticleFilter::PiggyBackParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>* create(B& m,
      Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs,
        out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>::PiggyBackParticleFilter(
    B& m, Random& rng, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs, out),
    ukf(m, rng, in, obs, NULL) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class R>
void bi::PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam) {
  /* pre-condition */
  assert (T > this->state.t);

  typedef typename locatable_vector<L,real>::type V1;
  typedef typename locatable_matrix<L,real>::type M1;

  int n = 0, r = 0;
  const int ND = this->m.getNetSize(D_NODE);
  const int NC = this->m.getNetSize(C_NODE);
  const int NR = this->m.getNetSize(R_NODE);
  const int NO = this->m.getNetSize(O_NODE);

  BOOST_AUTO(lws, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  ExpGaussianPdf<V1,M1> corrected(ND + NC + NR);
  ExpGaussianPdf<V1,M1> uncorrected(ND + NC + NR);
  ExpGaussianPdf<V1,M1> observed(NO);
  M1 SigmaXX(ND + NC + NR, ND + NC + NR);
  M1 SigmaXY1(ND + NC + NR, NO);
  M1 SigmaXY2(ND + NC + NR, NO);

  /* restore prior over initial state */
  subrange(corrected.mean(), 0, ND) = this->m.getPrior(D_NODE).mean();
  subrange(corrected.mean(), ND, NC) = this->m.getPrior(C_NODE).mean();
  subrange(corrected.mean(), ND + NC, NR).clear();
  corrected.cov().clear();
  subrange(corrected.cov(), 0, ND, 0, ND) = this->m.getPrior(D_NODE).cov();
  subrange(corrected.cov(), ND, NC, ND, NC) = this->m.getPrior(C_NODE).cov();

  matrix_scal(0.01, subrange(corrected.cov(), 0, ND + NC, 0, ND + NC));

  ident(subrange(corrected.cov(), ND + NC, NR, ND + NC, NR));
  corrected.init();

  init(theta, *lws, *as);
  while (this->state.t < T) {
    lookahead(s, *lws, corrected, observed, uncorrected, SigmaXX, SigmaXY1, SigmaXY2);
    predict(T, s);
    correct(s, *lws);
    output(n, s, r, *lws, *as);
    ++n;
    r = this->state.t < T; // no need to resample at last time
    if (r) {
      resample(s, *lws, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lws;
  delete as;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1, class R>
void bi::PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam) {
  /* pre-condition */
  assert (T > this->t);
  assert (this->out != NULL);

  int n = 0, r = 0, a;

  BOOST_AUTO(lws, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));
  synchronize();

  init(theta, *lws, *as);
  while (this->t < T) {
    lookahead(T, s, *lws);
    predict(T, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_NODE), 0) = column(xd, n);
    row(s.get(C_NODE), 0) = column(xc, n);
    row(s.get(R_NODE), 0) = column(xr, n);

    correct(s, *lws);
    output(n, s, r, *lws, *as);
    ++n;
    r = this->t < T; // no need to resample at last time
    if (r) {
      resample(s, a, *lws, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lws;
  delete as;
}

/**
 * @todo Currently assumes only one day between observations.
 */
template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class M2>
void bi::PiggyBackParticleFilter<B,IO1,IO2,IO3,CL>::lookahead(State<L>& s,
    V1& lws, ExpGaussianPdf<V2,M2>& corrected,
    ExpGaussianPdf<V2,M2>& observed, ExpGaussianPdf<V2,M2>& uncorrected,
    M2& SigmaXX, M2& SigmaXY1, M2& SigmaXY2) {
  const int P = s.size();
  const int ND = this->m.getNetSize(D_NODE);
  const int NC = this->m.getNetSize(C_NODE);
  const int NR = this->m.getNetSize(R_NODE);

  ExpGaussianPdf<V2,M2> pU(NR);
  ExpGaussianPdf<V2,M2> qU(NR);
  ExpGaussianPdf<V2,M2> pZ(ND + NC);
  ExpGaussianPdf<V2,M2> pX(ND + NC + NR);

  M2 C(NR, ND + NC);
  M2 Z(P, ND + NC);
  M2 X(P, ND + NC + NR);
  M2 K(NR, ND + NC);
  M2 U(P, NR);
  V2 lqs(P), lps(P), lws1(P), ws1(P), lxs(P);
  State<ON_HOST> s2(this->m, 0);

  const real to = this->oyUpdater.getTime();
  if (to >= this->state.t) {
    /* store current state */
    lws1 = lws;
    this->mark();

    element_exp(lws1.begin(), lws1.end(), ws1.begin());
    real Wt = 0.0;
    Wt = sum(ws1.begin(), ws1.end(), Wt);

//    if (this->state.t > 10) {
      columns(X, 0, ND) = s.get(D_NODE);
      columns(X, ND, NC) = s.get(C_NODE);
      columns(X, ND + NC, NR) = s.get(R_NODE);
      logCols(columns(X, 0, ND), this->m.getPrior(D_NODE).getLogs());
      logCols(columns(X, ND, NC), this->m.getPrior(C_NODE).getLogs());

      mean(X, ws1, corrected.mean(), Wt);
//      cov(X, ws1, corrected.mean(), corrected.cov(), Wt);
      corrected.init();
//    }

    /* auxiliary simulation forward */
    ukf.predict(to, corrected, s2, observed, uncorrected, SigmaXX, SigmaXY1, SigmaXY2);
    ukf.mark();
    ukf.correct(s2, corrected, observed, SigmaXY2, pX);
    pZ.mean() = subrange(pX.mean(), 0, ND + NC);
    pZ.cov() = subrange(pX.cov(), 0, ND + NC, 0, ND + NC);
    pZ.init();
    qU.mean() = subrange(pX.mean(), ND + NC, NR);
    qU.cov() = subrange(pX.cov(), ND + NC, NR, ND + NC, NR);
    qU.init();
    transpose(subrange(pX.cov(), 0, ND + NC, ND + NC, NR), C);
    ukf.restore();
    ukf.correct(s2, uncorrected, observed, SigmaXY1, corrected);

    std::cerr << "mean= " << std::endl;
    for (int i = 0; i < qU.mean().size(); ++i) {
        std::cerr << qU.mean()(i) << ' ';
    }
    std::cerr << std::endl;

    /* condition */
    columns(Z, 0, ND) = s.get(D_NODE);
    columns(Z, ND, NC) = s.get(C_NODE);
    logCols(columns(Z, 0, ND), this->m.getPrior(D_NODE).getLogs());
    logCols(columns(Z, ND, NC), this->m.getPrior(C_NODE).getLogs());
    sub_rows(Z, pZ.mean());
    set_rows(U, qU.mean());
    symm(1.0, pZ.prec(), C, 0.0, K, 'R', 'U');
    gemm(1.0, Z, K, 1.0, U, 'N', 'T');
    trsm(1.0, pZ.std(), C, 'R', 'U');
    syrk(-1.0, C, 1.0, qU.cov(), 'U');
    qU.mean().clear();
    qU.init();

    std::cerr << "cov= " << std::endl;
    for (int i = 0; i < qU.cov().size1(); ++i) {
      for (int j = 0; j < qU.cov().size2(); ++j) {
        std::cerr << qU.cov()(i,j) << ' ';
      }
      std::cerr << std::endl;
    }

//    for (int j = 0; j < qU.mean().size(); ++j) {
//      std::cerr << qU.mean()(j) << ' ';
//    }
//    std::cerr << std::endl;

    //if (this->state.t > 10) {
//    qU = pU;
      qU.samples(this->rng, s.get(R_NODE));
      qU.logDensities(s.get(R_NODE), lqs);
      matrix_axpy(1.0, U, s.get(R_NODE));
      pU.logDensities(s.get(R_NODE), lps);

      /**
       * Correct weights; let \f$p^i = p(\mathbf{u}^i)\f$ and
       * \f$q^i = q(\mathbf{u}^i)\f$, then:
       *
       * \f[\ln \mathbf{w} \gets \ln \mathbf{w} + \ln \mathbf{p} -
       * \ln \mathbf{q}.\f]
       */
      axpy(1.0, lps, lws);
      axpy(-1.0, lqs, lws);

      /* restore previous state */
      this->rUpdater.skipNext();
    //}
    this->restore();

    /* post-condition */
    assert (this->sim.getTime() == this->getTime());
  }
}

#endif
