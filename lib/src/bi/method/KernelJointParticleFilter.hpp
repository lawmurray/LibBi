/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1246 $
 * $Date: 2011-01-31 16:23:46 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP
#define BI_METHOD_DISTURBANCEPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"
#include "../updater/OUpdater.hpp"
#include "../updater/ORUpdater.hpp"

namespace bi {
/**
 * Disturbance particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::DisturbanceParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class DisturbanceParticleFilter : public ParticleFilter<B,IO1,IO2,IO3,CL> {
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
  DisturbanceParticleFilter(B& m, Random& rng, IO1* in = NULL,
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
  template<Location L, class V1, class V2, class R>
  void lookahead(State<L>& s, V1& lws, V2& as, R* resam);
  //@}

private:
  /**
   * O-net updater.
   */
  OUpdater<B> oUpdater;

  /**
   * Or-net updater.
   */
  ORUpdater<B> orUpdater;
};

/**
 * Factory for creating DisturbanceParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see DisturbanceParticleFilter
 */
template<Location CL = ON_HOST>
struct DisturbanceParticleFilterFactory {
  /**
   * Create disturbance particle filter.
   *
   * @return DisturbanceParticleFilter object. Caller has ownership.
   *
   * @see DisturbanceParticleFilter::DisturbanceParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>* create(B& m,
      Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs,
        out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::DisturbanceParticleFilter(
    B& m, Random& rng, IO1* in, IO2* obs, IO3* out) :
    ParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs, out),
    orUpdater(rng) {
  //
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam) {
  /* pre-condition */
  assert (T > this->state.t);

  int n = 0, r = 0;

  BOOST_AUTO(lws, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  init(theta, *lws, *as);
  while (this->state.t < T) {
    lookahead(s, *lws, *as, resam);
    predict(T, s);
    correct(s, *lws);
    output(n, s, r, *lws, *as);
    ++n;
//    r = true;
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
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
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

#include <fstream>

/**
 * @todo Currently assumes only one day between observations.
 */
template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::DisturbanceParticleFilter<B,IO1,IO2,IO3,CL>::lookahead(State<L>& s,
    V1& lws, V2& as, R* resam) {
  typedef typename locatable_temp_vector<L,real>::type temp_vector_type;
  typedef typename locatable_temp_matrix<L,real>::type temp_matrix_type;

  const real to = this->oyUpdater.getTime();
  if (to >= this->state.t) {
    const int P = s.size();
    const int ND = this->m.getNetSize(D_NODE);
    const int NC = this->m.getNetSize(C_NODE);
    const int NR = this->m.getNetSize(R_NODE);

    /**
     * Let \f$\{ \mathbf{z}^i \equiv (\mathbf{x}^i_{t-1}, \mathbf{y}^i_t) \}\f$,
     * for \f$i = 1,\ldots,P\f$, be the set of state (d-node and c-node)
     * particles at time \f$t - 1\f$, augmented by drawing a predicted
     * observation (o-node) at time \f$t\f$.
     *
     * Approximate \f$p(Z)\f$ by a (log-)Gaussian with mean vector
     * \f$\boldsymbol{\mu}_Z\f$ and covariance matrix \f$\Sigma_Z\f$.
     *
     * Let \f$\{ \mathbf{u}^i \equiv \mathbf{u}^i_t \}\f$, for
     * \f$i = 1,\ldots,P\f$, be the set of disturbance particles (r-node)
     * for time \f$t\f$. \f$p(U)\f$ is a (log-)Gaussian with zero
     * mean and identity covariance. We wish to compute a proposal
     * distribution \f$q(U) \approx p(U|Z)\f$, to be approximated by a
     * (log-)Gaussian with mean vector \f$\boldsymbol{\mu}_U\f$ and covariance
     * matrix \f$\Sigma_U\f$. Let \f$C\f$ be the sample cross-covariance
     * matrix between \f$U\f$ and \f$Z\f$ from the initial propagation forward
     * to compute predicted observations, where each
     * \f$\mathbf{u}^i \sim p(U)\f$.
     */
    temp_matrix_type X(P, ND + NC);
    temp_matrix_type Y(P, NR), S(P, NR);
    temp_matrix_type Z(P, ND + NC);
    temp_matrix_type C(NR, ND + NC);
    temp_matrix_type K(NR, ND + NC);
    temp_matrix_type U(P, NR);
    temp_matrix_type D(P, P);
    temp_vector_type lqs(P), lps(P), lws1(P), ws1(P), lxs(P), z(P), w(P);
    lws1 = lws;

    ExpGaussianPdf<temp_vector_type,temp_matrix_type> pU(NR);
    ExpGaussianPdf<temp_vector_type,temp_matrix_type> qU(NR);
    ExpGaussianPdf<temp_vector_type,temp_matrix_type> pZ(ND + NC);

    columns(X, 0, ND) = s.get(D_NODE);
    columns(X, ND, NC) = s.get(C_NODE);
    columns(Z, 0, ND) = s.get(D_NODE);
    columns(Z, ND, NC) = s.get(C_NODE);

    /* store current state */
    this->mark();

    /* auxiliary simulation forward */
    this->predict(to, s);
    this->correct(s, lws1);

    logCols(columns(Z, 0, ND), this->m.getPrior(D_NODE).getLogs());
    logCols(columns(Z, ND, NC), this->m.getPrior(C_NODE).getLogs());
    element_exp(lws1.begin(), lws1.end(), w.begin());

    /* auxiliary resample */
//    resam->resample(lws1, lws, as);
//    resam->copy(as, X);
//    resam->copy(as, Z);

    /* compute distances */
    mean(Z, w, pZ.mean());
    cov(Z, w, pZ.mean(), pZ.cov());
    pZ.init();
    standardise(pZ, Z);
    distance(Z, 1.0, D); // hopt(ND + NC, P)
    symv(1.0, D, w, 0.0, z, 'U'); // normalisation terms
    element_rcp(z.begin(), z.end(), z.begin());

    /* compute approximate transition kernel means */
    gdmm(1.0, w, s.get(R_NODE), 0.0, U);
    symm(1.0, D, U, 0.0, Y, 'L', 'U');
    gdmm(1.0, z, Y, 0.0, U); // means now in U

    /* compute kernel standard deviations (orthogonal) */
    Y = s.get(R_NODE);
    matrix_axpy(-1.0, U, Y);
    element_square(Y.begin(), Y.end(), Y.begin());

    gdmm(1.0, w, Y, 0.0, S);
    symm(1.0, D, S, 0.0, Y, 'L', 'U');
    gdmm(1.0, z, Y, 0.0, S); // std. devs. now in S

    //bi::fill(S.begin(), S.end(), 1.0);

//    std::ofstream out("U.csv");
//    for (int i = 0; i < U.size1(); ++i) {
//      for (int j = 0; j < U.size2(); ++j) {
//        out << U(i,j) << ' ';
//      }
//      out << std::endl;
//    }
//    std::ofstream out2("S.csv");
//    for (int i = 0; i < S.size1(); ++i) {
//      for (int j = 0; j < S.size2(); ++j) {
//        out2 << S(i,j) << ' ';
//      }
//      out2 << std::endl;
//    }

    /* samples */
    pU.samples(this->rng, s.get(R_NODE));
    for (int i = 0; i < s.get(R_NODE).size1(); ++i) {
      BOOST_AUTO(x, row(s.get(R_NODE),i));
      BOOST_AUTO(mu, row(U,i));
      BOOST_AUTO(sigma, row(S,i));

      lqs(i) = -0.5*dot(x,x) - NR*BI_HALF_LOG_TWO_PI - 0.5*log(prod(sigma.begin(), sigma.end(), 1.0));
      gdmv(1.0, sigma, x, 1.0, mu);
    }
    pU.logDensities(U, lps);
    s.get(R_NODE) = U;

//        std::ofstream out2("lqs.csv");
//          for (int j = 0; j < lqs.size(); ++j) {
//            out2 << lqs(j) << ' ';
//          }
//          out2 << std::endl;
//          std::ofstream out3("lps.csv");
//            for (int j = 0; j < lps.size(); ++j) {
//              out3 << lps(j) << ' ';
//            }
//            out3 << std::endl;

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
    s.get(D_NODE) = columns(X, 0, ND);
    s.get(C_NODE) = columns(X, ND, NC);
    this->rUpdater.skipNext();
    this->restore();

    /* post-condition */
    assert (this->sim.getTime() == this->getTime());
  }
}

#endif
