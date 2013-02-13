/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SMC2_HPP
#define BI_METHOD_SMC2_HPP

#include "misc.hpp"
#include "ParticleMarginalMetropolisHastings.hpp"
#include "../state/ThetaParticle.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../pdf/misc.hpp"
#include "../pdf/GaussianPdf.hpp"

namespace bi {
/**
 * @internal
 *
 * State of SMC2.
 */
struct SMC2State {
  /**
   * Constructor.
   */
  SMC2State();

  /**
   * Current time.
   */
  real t;
};
}

bi::SMC2State::SMC2State() :
    t(0.0) {
  //
}

namespace bi {
/**
 * Sequential Monte Carlo squared (SMC^2).
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam F #concept::Filter type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 #concept::SMC2Buffer type.
 */
template<class B, class F, class R, class IO1>
class SMC2 {
public:
  /**
   * Constructor.
   *
   * @tparam IO2 Input type.
   *
   * @param m Model.
   * @param filter Filter.
   * @param resam Resampler for theta-particles.
   * @param out Output.
   *
   * @see ParticleFilter
   */
  SMC2(B& m, F* pmmh = NULL, R* resam = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   */
  //@{
  /**
   * Get filter.
   *
   * @return Filter.
   */
  F* getFilter();

  /**
   * Set filter.
   *
   * @param filter Filter.
   */
  void setFilter(F* pmmh);

  /**
   * Get resampler.
   *
   * @return Resampler.
   */
  R* getResam();

  /**
   * Set resampler.
   *
   * @param resam Resampler.
   */
  void setResam(R* resam);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output buffer.
   */
  void setOutput(IO1* out);

  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param s Prototype state. The set of \f$\theta\f$-particles are
   * constructed from this.
   * @param inInit Initialisation file.
   * @param D Number of samples to draw.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling of theta-particles.
   * @param localMove Is this a local proposal?
   * @param Nmoves Number of steps per \f$\theta\f$-particle.
   * @param moveScale Scaling factor for proposal.
   *
   */
  template<Location L, class IO2>
  void sample(Random& rng, const real t, const real T, const int K,
      ThetaParticle<B,L>& s, IO2* inInit = NULL, const int D = 1,
      const real essRel = 0.5, const int localMove = 0, const int Nmoves = 1,
      const real moveScale = 0.5);
  //@}

  /**
   * @name Low-level interface.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param t Start time.
   * @param s Prototype state. The set of \f$\theta\f$-particles are
   * constructed from this.
   * @param thetas[out] The set of \f$\theta\f$-particles.
   * @param lws[out] Log-weights of \f$\theta\f$-particles.
   * @param as[out] Ancestors of \f$\theta\f$-particles.
   */
  template<Location L, class V1, class V2>
  void init(Random& rng, const real t, ThetaParticle<B,L>& s,
      std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, V2 as);

  /**
   * Step all x-particles forward.
   *
   * @return Evidence.
   */
  template<Location L, class V1>
  real step(Random& rng, const real T,
      std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, const int n);

  /**
   * Adapt proposal distribution.
   */
  template<Location L, class V1, class V2, class M2>
  void adapt(std::vector<ThetaParticle<B,L>*>& thetas, V1 lws,
      const bool localMove, GaussianPdf<V2,M2>& q);

  /**
   * Resample \f$\theta\f$-particles.
   */
  template<Location L, class V1, class V2>
  void resample(Random& rng, V1 lws, V2 as,
      std::vector<ThetaParticle<B,L>*>& thetas);

  /**
   * Rejuvenate \f$\theta\f$-particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param s Working state.
   * @param thetas[out] The set of \f$\theta\f$-particles.
   * @param lws[out] Log-weights of \f$\theta\f$-particles.
   * @param q Proposal distribution.
   * @param localMove Is this a local proposal?
   * @param Nmoves Number of steps per \f$\theta\f$-particle.
   * @param moveScale Scaling factor for proposal.
   *
   * @return Acceptance rate.
   */
  template<Location L, class V1, class V2, class M2>
  real rejuvenate(Random& rng, const real t, const real T, const int K,
      ThetaParticle<B,L>& s, const std::vector<ThetaParticle<B,L>*>& thetas,
      const V1 lws, GaussianPdf<V2,M2>& q, const bool localMove,
      const int Nmoves, const real moveScale);

  /**
   * Output.
   *
   * @param n Step number.
   * @param thetas Theta-particles.
   * @param lws Log-weights of theta-particles.
   * @param evidence Evidence.
   * @param ess Effective sample size of theta-particles.
   * @param acceptRate Acceptance rate of rejuvenation step
   */
  template<bi::Location L, class V1>
  void output(const int n, const std::vector<ThetaParticle<B,L>*>& thetas,
      const V1 lws, const real evidence, const real ess,
      const real acceptRate);

  /**
   * Report progress on stderr.
   *
   * @param n Step number.
   * @param t Time.
   * @param ess Effective sample size of theta-particles.
   * @param r Was resampling performed?
   * @param acceptRate Acceptance rate of rejuvenation step (if any).
   */
  void report(const int n, const real t, const real ess, const bool r,
      const real acceptRate);

  /**
   * Terminate.
   */
  void term();
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * PMCMC sampler.
   */
  F* pmmh;

  /**
   * Resampler for the theta-particles
   */
  R* resam;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Effective Sample Size threshold:
   * when the theta-particles' ESS goes below this value,
   * a resample-move step is triggered
   */
  real essRel;

  /**
   * State.
   */
  SMC2State state;

  /* net sizes, for convenience */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NP = B::NP;
};

/**
 * Factory for creating SMC2 objects.
 *
 * @ingroup method
 *
 * @see SMC2
 */
struct SMC2Factory {
  /**
   * Create particle MCMC sampler.
   *
   * @return SMC2 object. Caller has ownership.
   *
   * @see SMC2::SMC2()
   */
  template<class B, class F, class R, class IO1>
  static SMC2<B,F,R,IO1>* create(B& m, F* pmmh = NULL, R* resam = NULL,
      IO1* out = NULL) {
    return new SMC2<B,F,R,IO1>(m, pmmh, resam, out);
  }
};
}

#include "../math/misc.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

#include "boost/typeof/typeof.hpp"

template<class B, class F, class R, class IO1>
bi::SMC2<B,F,R,IO1>::SMC2(B& m, F* pmmh, R* resam, IO1* out) :
    m(m), pmmh(pmmh), resam(resam), out(out) {
  //
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class IO2>
void bi::SMC2<B,F,R,IO1>::sample(Random& rng, const real t, const real T,
    const int K, ThetaParticle<B,L>& s, IO2* inInit, const int D,
    const real essRel, const int localMove, const int Nmoves,
    const real moveScale) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  typedef typename loc_temp_vector<L,real>::type logweight_vector_type;
  typedef typename loc_temp_vector<L,int>::type ancestor_vector_type;

  typedef typename temp_host_vector<real>::type host_logweight_vector_type;
  typedef typename temp_host_vector<int>::type host_ancestor_vector_type;

  this->essRel = essRel;

  std::vector<ThetaParticle<B,L>*> thetas(D);  // theta-particles
  host_vector_type lws(D);  // log-weights of theta-particles
  host_ancestor_vector_type as(D);  // ancestors
  real evidence = 0.0, ess = 0.0, acceptRate = -1.0;
  GaussianPdf<host_vector_type,host_matrix_type> q(NP);  // proposal distro

  /* init */
  init(rng, t, s, thetas, lws, as);

  /* sample */
  int k = 0, n = 0;
  real tk;
  bool r = false;  // resampling performed?
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      evidence = step(rng, tk, thetas, lws, n);
      ess = resam->ess(lws);
      r = ess < D * essRel;
      if (r) {
        /* resample-move */
        adapt(thetas, lws, localMove, q);
        resample(rng, lws, as, thetas);
        acceptRate = rejuvenate(rng, t, state.t, k, s, thetas, lws, q,
            localMove, Nmoves, moveScale);
      } else {
        acceptRate = 0.0;
      }
      report(n, state.t, ess, r, acceptRate);
      output(n, thetas, lws, evidence, ess, acceptRate);
      ++n;
    } while (state.t < tk);

    ++k;
  } while (k <= K);

  /* delete remaining stuff */
  for (int i = 0; i < D; i++) {
    delete thetas[i];
  }

  /* terminate */
  term();
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::SMC2<B,F,R,IO1>::init(Random& rng, const real t,
    ThetaParticle<B,L>& s, std::vector<ThetaParticle<B,L>*>& thetas, V1 lws,
    V2 as) {
  typename loc_temp_vector<L,real>::type logPrior(1);
  int i;

  /* set time */
  state.t = t;

  /* initialise theta-particles */
  for (i = 0; i < thetas.size(); ++i) {
    thetas[i] = new ThetaParticle<B,L>(s.size(), s.getTrajectory().size2());
    ThetaParticle<B,L>& theta = *thetas[i];

    m.parameterSample(rng, theta);
    theta.get(PY_VAR) = theta.get(P_VAR);
    theta.getLogPrior1() = m.parameterLogDensity(theta);

    /* initialise x-particles for this theta */
    pmmh->getFilter()->init(rng, t, row(theta.get(P_VAR), 0), theta,
        theta.getLogWeights(), theta.getAncestors());
  }

  /* initialise log-weights and ancestors */
  lws.clear();
  seq_elements(as, 0);
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1>
real bi::SMC2<B,F,R,IO1>::step(Random& rng, const real T,
    std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, const int n) {
  int i, r;
  real evidence = 0.0;

  for (i = 0; i < thetas.size(); i++) {
    BOOST_AUTO(&s, *thetas[i]);
    BOOST_AUTO(&thetaLws, s.getLogWeights());
    BOOST_AUTO(&thetaAs, s.getAncestors());
    BOOST_AUTO(&thetaIncLl, s.getIncLogLikelihood());
    BOOST_AUTO(&thetaLl, s.getLogLikelihood1());

    /* set up filter */
    if (i == 0) {
      pmmh->getFilter()->getSim()->setTime(state.t, s);  // may have been left inconsistent from last rejuvenate
      pmmh->getFilter()->mark();
    } else {
      pmmh->getFilter()->top();
    }
    r = pmmh->getFilter()->step(rng, T, s, thetaLws, thetaAs, n);
    pmmh->getFilter()->output(n, s, r, thetaLws, thetaAs);

    thetaIncLl = logsumexp_reduce(thetaLws)
        - bi::log(static_cast<real>(thetaLws.size()));
    thetaLl += thetaIncLl;
    lws(i) += thetaIncLl;

    /* compute evidence along the way */
    evidence += bi::exp(lws(i) + thetaIncLl);
  }
  pmmh->getFilter()->pop();
  state.t = pmmh->getFilter()->getSim()->getTime();

  evidence /= sumexp_reduce(lws);

  return evidence;
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2, class M2>
void bi::SMC2<B,F,R,IO1>::adapt(std::vector<ThetaParticle<B,L>*>& thetas,
    V1 lws, const bool localMove, GaussianPdf<V2,M2>& q) {
  typedef typename sim_temp_vector<V2>::type vector_type;
  typedef typename sim_temp_matrix<M2>::type matrix_type;

  const int P = lws.size();
  vector_type ws(P), mu(NP);
  matrix_type Sigma(NP, NP), U(NP, NP), X(P, NP);
  int p;

  /* copy parameters into matrix */
  for (p = 0; p < P; ++p) {
    row(X, p) = row(thetas[p]->get(P_VAR), 0);
  }

  /* compute weighted mean, covariance and Cholesky factor */
  expu_elements(lws, ws);
  mean(X, ws, mu);
  cov(X, ws, mu, Sigma);
  chol(Sigma, U);

  /* write proposal */
  if (!localMove) {
    q.setMean(mu);
  } else {
    q.mean().clear();
  }
  q.setStd(U);
  q.init();
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::SMC2<B,F,R,IO1>::resample(Random& rng, V1 lws, V2 as,
    std::vector<ThetaParticle<B,L>*>& thetas) {
  resam->resample(rng, lws, as, thetas);
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2, class M2>
real bi::SMC2<B,F,R,IO1>::rejuvenate(Random& rng, const real t, const real T,
    const int K, ThetaParticle<B,L>& s,
    const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    GaussianPdf<V2,M2>& q, const bool localMove, const int Nmoves,
    const real moveScale) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  const int P = thetas.size();

  host_matrix_type thetaparticles(P, NP);  // current parameters
  host_matrix_type thetaproposals(P, NP);  // proposed moves
  host_vector_type logdensCurrent(P);  // reverse proposal log-densities
  host_vector_type logdensProposals(P);  // proposal log-densities

  int p;
  for (p = 0; p < P; ++p) {
    row(thetaparticles, p) = row(thetas[p]->get(P_VAR), 0);
  }

  double meanAcceptRate = 0.;
  for (int indexMove = 0; indexMove < Nmoves; indexMove++) {
    q.samples(rng, thetaproposals);
    if (!localMove) {
      q.logDensities(thetaproposals, logdensProposals, true);
      q.logDensities(thetaparticles, logdensCurrent, true);
    } else {
      matrix_scal(moveScale, thetaproposals);
      matrix_axpy(1.0, thetaparticles, thetaproposals, false);
      logdensProposals.clear();  // symmetric proposal
      logdensCurrent.clear();
    }

    int sumaccept = 0;
    for (p = 0; p < P; ++p) {
      ThetaParticle<B,L>& theta = *thetas[p];

      theta.getParameters1() = row(thetaparticles, p);
      s.resize(theta.size());
      s = theta;
      bool pmmhaccept = true;

      /* instead of using PMMH->propose, we use the Gaussian density fitted
       * on the current theta particles to propose new theta particles */
      //s.getParameters2() = row(thetaproposals, p);
      //s.getLogProposal1() = logdensCurrent[p];
      //s.getLogProposal2() = logdensProposals[p];

      pmmh->propose(rng, s);
      pmmh->logPrior(s);
      pmmh->logLikelihood(rng, t, T, K, s);
      pmmhaccept = pmmh->computeAcceptReject(rng, s);
      if (pmmhaccept) {
        pmmh->accept(rng, s);

        // NB: ancestors are not copied as they have already been copied earlier.
        row(thetaparticles, p) = s.getParameters1();
        s.getLogWeights().resize(s.size());
        s.getLogWeights() = pmmh->getFilter()->getOutput()->getLogWeights();

        theta.resize(s.size());
        theta = s;
      } else {
        pmmh->reject();
      }
      sumaccept += pmmhaccept;
    }
    meanAcceptRate += (double)sumaccept / (double)thetas.size();
  }
  return meanAcceptRate / (double)Nmoves;
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1>
void bi::SMC2<B,F,R,IO1>::output(const int n,
    const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    const real evidence, const real ess, const real acceptRate) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  const int P = thetas.size();

  host_vector_type allNx(P);
  host_matrix_type X(P, NP);

  int p;

  /* copy parameters into matrix */
  for (p = 0; p < P; ++p) {
    row(X, p) = row(thetas[p]->get(P_VAR), 0);
    allNx(p) = thetas[p]->size();
  }

  out->writeState(P_VAR, n, X);
  out->writeLogWeights(n, lws);
  out->writeNumberX(n, allNx);
  out->writeEvidence(n, evidence);
  out->writeEss(n, ess);
  out->writeAcceptanceRate(n, acceptRate);
}

template<class B, class F, class R, class IO1>
void bi::SMC2<B,F,R,IO1>::report(const int n, const real t, const real ess,
    const bool r, const real acceptRate) {
  std::cerr << n << ":\ttime " << t << "\tESS " << ess;
  if (r) {
    std::cerr << "\tresample-move with acceptance rate " << acceptRate;
  }
  std::cerr << std::endl;
}

template<class B, class F, class R, class IO1>
void bi::SMC2<B,F,R,IO1>::term() {
  //
}

#endif
