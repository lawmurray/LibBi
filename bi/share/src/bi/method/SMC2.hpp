/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2597 $
 * $Date: 2012-05-24 14:33:24 +0000 (Thu, 24 May 2012) $
 */
#ifndef BI_METHOD_SMC2_HPP
#define BI_METHOD_SMC2_HPP

#include "misc.hpp"
#include "ThetaParticle.hpp"
#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/io.hpp"
#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../pdf/misc.hpp"
#include "../pdf/GaussianPdf.hpp"
#include "ParticleMarginalMetropolisHastings.hpp"

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

bi::SMC2State::SMC2State() : t(0.0) {
  //
}

namespace bi {
/**
 * Sequential Monte Carlo square (SMC^2)
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam R #concept::Resampler type.
 * @tparam IO1 #concept::SMC2Buffer type.
 * @tparam CL Cache location.
 */
template<class B, class R, class IO1, Location CL = ON_HOST>
class SMC2 {
public:
  /**
   * Constructor.
   *
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param m Model.
   * @param resam Resampler for theta-particles.
   * @param out Output.
   *
   * @see ParticleFilter
   */
  SMC2(B& m, R* resam = NULL, IO1* out = NULL);

  /**
   * Get output buffer.
   */
  IO1* getOutput();

  /**
   * @name High-level interface.
   */
  //@{
  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Length of time to sample over.
   * @param s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   * @param D Number of samples to draw.
   * @param Nm
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling of theta-particles.
   */
  template<Location L, class F, class IO2>
  void sample(Random& rng, const real T, State<B,L>& s, F* filter,
      IO2* inInit = NULL, const int D = 1, const int Nm = 1,
      const real essRel = 0.5, const int localMove = 0,
      const real moveScale = 0.5);
  //@}

  /**
   * @name Low-level interface.
   */
  //@{
  /**
   * Initialise.
   */
  template<Location L, class F, class V1, class V2>
  void init(Random& rng, std::vector<ThetaParticle<B,L>*>& thetas, F* filter,
      V1 lws, V2 as);

  /**
   * Step all x-particles forward.
   *
   * @return Evidence.
   */
  template<Location L, class V1, class F>
  real step(Random& rng, const real T,
      std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, F* filter,
      const int n);

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
   * @return Acceptance rate.
   */
  template<Location L, class F1, class F2, class V1, class V2, class M2>
  real rejuvenate(Random& rng, State<B,L>& s_in, F1* filter,
      F2* pmmh, const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
      GaussianPdf<V2,M2>& q, const bool localMove, const real moveScale);

  /**
   * Output.
   */
  template<bi::Location L, class V1>
  void output(const int n, const std::vector<ThetaParticle<B,L>*>& thetas,
      const V1 lws, const real evidence, const real ess,
      const real acceptanceRate);
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Number of theta-particles
   */
  int Ntheta;

  /**
   * Number of x-particles
   */
  int Nx;

  /**
   * Number of PMMH moves performed at each resample-move step
   */
  int Nmoves;

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

  /**
   * Resampler for the theta-particles
   */
  R* resam;

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
};

/**
 * Factory for creating SMC2 objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see SMC2
 */
template<Location CL = ON_HOST>
struct SMC2Factory {
  /**
   * Create particle MCMC sampler.
   *
   * @return SMC2 object. Caller has ownership.
   *
   * @see SMC2::SMC2()
   */
  template<class B, class R, class IO1>
  static SMC2<B,R,IO1,CL>* create(B& m, R* resam = NULL, IO1* out = NULL) {
    return new SMC2<B,R,IO1,CL>(m, resam, out);
  }
};
}

#include "../math/misc.hpp"
#include "../math/io.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

#include <netcdfcpp.h>

#include "boost/typeof/typeof.hpp"

template<class B, class R, class IO1, bi::Location CL>
bi::SMC2<B,R,IO1,CL>::SMC2(B& m, R* resam, IO1* out) :
    m(m),
    resam(resam),
    out(out) {
  //
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::SMC2<B,R,IO1,CL>::sample(Random& rng, const real T, State<B,L>& s_in,
    F* filter, IO2* inInit, const int D, const int Nm, const real essRel,
    const int localMove, const real moveScale) {
  /**
   * @todo Add the possibility to output the theta particles at different
   * time steps
   */
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  typedef typename loc_temp_vector<L,real>::type logweight_vector_type;
  typedef typename loc_temp_vector<L,int>::type ancestor_vector_type;

  typedef typename temp_host_vector<real>::type host_logweight_vector_type;
  typedef typename temp_host_vector<int>::type host_ancestor_vector_type;

  this->Nx = s_in.size();
  this->Ntheta = D;
  this->Nmoves = Nm;
  this->essRel = essRel;
  BOOST_AUTO(pmmh, ParticleMarginalMetropolisHastingsFactory<CL>::create(m,
      out));
  std::vector<ThetaParticle<B,L>*> thetas(Ntheta); // theta-particles
  host_vector_type lws(Ntheta); // log-weights of theta-particles
  host_ancestor_vector_type as(Ntheta); // ancestors
  real evidence = 0.0, ess = 0.0, acceptanceRate = -1.0;
  GaussianPdf<host_vector_type,host_matrix_type> q(NP); // proposal distro

  /* init */
  init(rng, thetas, filter, lws, as);

  /* sample */
  int n = 0, r = 0;
  while (state.t < T) {
    std::cerr << "time " << state.t << std::endl;
    evidence = step(rng, T, thetas, lws, filter, n);
    ess = resam->ess(lws);
    std::cerr << "ESS at time " << state.t << ": " << ess << std::endl;
    if (ess < Ntheta*essRel) {
      /* resample-move step */
      std::cerr << "Resampling step at time " << state.t
          << " (threshold: " << Ntheta*essRel << ")" << std::endl;
      adapt(thetas, lws, localMove, q);
      resample(rng, lws, as, thetas);
      acceptanceRate = rejuvenate(rng, s_in, filter, pmmh, thetas, lws, q,
          localMove, moveScale);
    } else {
      acceptanceRate = -1.0;
    }
    output(n, thetas, lws, evidence, ess, acceptanceRate);
    ++n;
  }

  /* delete remaining stuff */
  delete pmmh;
  for (int i = 0; i < Ntheta; i++) {
    delete thetas[i];
  }
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class F, class V1, class V2>
void bi::SMC2<B,R,IO1,CL>::init(Random& rng,
    std::vector<ThetaParticle<B,L>*>& thetas, F* filter, V1 lws, V2 as) {
  typename loc_temp_vector<L,real>::type logPrior(1);
  int i;

  /* set time */
  state.t = 0.0;

  /* initialise theta-particles */
  for (i = 0; i < Ntheta; ++i) {
    thetas[i] = new ThetaParticle<B,L>(m, Nx);
    ThetaParticle<B,L>& theta = *thetas[i];
    State<B,L>& s = theta.getState();

    logPrior.clear();
    s.setRange(0, 1);
    m.parameterSamples(rng, s);
    s.getAlt(P_VAR) = s.get(P_VAR);
    m.parameterLogDensities(s, logPrior);
    theta.getLogPrior() = *logPrior.begin();
    s.setRange(0, Nx);
  }

  /* initialise x-particles */
  for (i = 0; i < Ntheta; ++i) {
    ThetaParticle<B,L>& theta = *thetas[i];
    State<B,L>& s = theta.getState();

    filter->init(rng, row(s.get(P_VAR), 0), s, theta.getLogWeights(),
        theta.getAncestors());
  }

  /* initialise log-weights and ancestors */
  lws.clear();
  seq_elements(as, 0);
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class V1, class F>
real bi::SMC2<B,R,IO1,CL>::step(Random& rng, const real T,
    std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, F* filter,
    const int n) {
  int i, r;
  real evidence = 0.0, tnxt = filter->getNextObsTime(T);

  for (i = 0; i < Ntheta; i++) {
    if (i == 0) {
      filter->mark();
    } else {
      filter->top();
    }
    ThetaParticle<B,L>& theta = *thetas[i];
    State<B,L>& s = theta.getState();

    filter->setTime(state.t, s);
    r = filter->step(rng, tnxt, s, theta.getLogWeights(), theta.getAncestors(), n);

    filter->output(n, s, r, theta.getLogWeights(), theta.getAncestors());
    theta.getIncLogLikelihood() = logsumexp_reduce(theta.getLogWeights()) -
        std::log(theta.getLogWeights().size());
    theta.getLogLikelihood() += theta.getIncLogLikelihood();
    lws(i) += theta.getIncLogLikelihood();

    /* compute evidence along the way */
    evidence += exp(lws(i))*exp(theta.getIncLogLikelihood());
  }
  filter->pop();
  state.t = tnxt;

  evidence /= sumexp_reduce(lws);

  return evidence;
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class V1, class V2, class M2>
void bi::SMC2<B,R,IO1,CL>::adapt(std::vector<ThetaParticle<B,L>*>& thetas,
    V1 lws, const bool localMove, GaussianPdf<V2,M2>& q) {
  typedef typename sim_temp_vector<V2>::type vector_type;
  typedef typename sim_temp_matrix<M2>::type matrix_type;

  vector_type ws(Ntheta), mu(NP);
  matrix_type Sigma(NP,NP), U(NP,NP), X(Ntheta, NP);
  int i;

  /* copy parameters into matrix */
  for (i = 0; i < Ntheta; ++i) {
    row(X, i) = row(thetas[i]->getState().get(P_VAR), 0);
  }

  /* compute weighted mean, covariance and Cholesky factor */
  ws = lws;
  expu_elements(ws);
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

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::SMC2<B,R,IO1,CL>::resample(Random& rng, V1 lws, V2 as,
    std::vector<ThetaParticle<B,L>*>& thetas) {
  resam->resample(rng, lws, as, thetas);
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class F1, class F2, class V1, class V2, class M2>
real bi::SMC2<B,R,IO1,CL>::rejuvenate(Random& rng, State<B,L>& s_in, F1* filter,
    F2* pmmh, const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    GaussianPdf<V2,M2>& q, const bool localMove, const real moveScale) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  host_matrix_type thetaparticles(Ntheta, NP); // current parameters
  host_matrix_type thetaproposals(Ntheta, NP); // proposed moves
  host_vector_type logdensCurrent(Ntheta); // reverse proposal log-densities
  host_vector_type logdensProposals(Ntheta); // proposal log-densities

  ParticleMarginalMetropolisHastingsState& x1 = pmmh->getState();
  ParticleMarginalMetropolisHastingsState& x2 = pmmh->getOtherState();

  for (int i = 0; i < Ntheta; ++i) {
    row(thetaparticles, i) = row(thetas[i]->getState().get(P_VAR), 0);
  }

  double meanacceptrate = 0.;
  for (int indexmove = 0; indexmove < Nmoves; indexmove++) {
    q.samples(rng, thetaproposals);
    if (!localMove) {
      q.logDensities(thetaproposals, logdensProposals, true);
      q.logDensities(thetaparticles, logdensCurrent, true);
    } else {
      matrix_scal(moveScale, thetaproposals);
      matrix_axpy(1.0, thetaparticles, thetaproposals, false);
      logdensProposals.clear(); // symmetric proposal
      logdensCurrent.clear();
    }

    int sumaccept = 0;
    for (int i = 0; i < Ntheta; ++i) {
      ThetaParticle<B,L>& theta = *thetas[i];
      State<B,L>& s = theta.getState();

      x1.theta = row(thetaparticles, i);
      x1.ll = theta.getLogLikelihood();
      x1.lp = theta.getLogPrior();
      s_in.resize(s.size());
      s_in = s;
      bool pmmhaccept = true;

      /* instead of using PMMH->propose, we use the Gaussian density fitted
       * on the current theta particles to propose new theta particles */
      x2.theta = row(thetaproposals, i);
      x1.lq = logdensCurrent[i];
      x2.lq = logdensProposals[i];
      pmmh->computePriorOnProposal(s_in);
      bool cpf_fail = false;
      if (!BI_IS_FINITE(x2.lp)) {
        pmmhaccept = false;
        x2.ll = -1.0/0.0;
      } else {
        cpf_fail = pmmh->estimateLLOnProposal(rng, state.t, s_in, filter);
      }
      pmmh->computeAcceptReject(rng, filter, cpf_fail, pmmhaccept);
      if (pmmhaccept) {
        // NB: ancestors are not copied as they have already been copied earlier.
        row(thetaparticles, i) = x1.theta;
        theta.getLogLikelihood() = x1.ll;
        theta.getLogPrior() = x1.lp;
        theta.getLogWeights().resize(s_in.size());
        theta.getLogWeights() = filter->getLogWeights();
        theta.getState().resize(s_in.size());
        theta.getState() = s_in;
      }
      sumaccept += pmmhaccept;
    }
    meanacceptrate += (double)sumaccept / (double)Ntheta;
    std::cerr << "accept rate: " << (double)sumaccept / (double)Ntheta<< std::endl;
  }

  return meanacceptrate / (double) Nmoves;
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class V1>
void bi::SMC2<B,R,IO1,CL>::output(const int n,
    const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    const real evidence, const real ess, const real acceptanceRate) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  host_vector_type allNx(Ntheta);
  host_matrix_type X(Ntheta, NP);
  int i;

  /* copy parameters into matrix */
  for (i = 0; i < Ntheta; ++i) {
    row(X, i) = row(thetas[i]->getState().get(P_VAR), 0);
    allNx(i) = thetas[i]->getState().size();
  }

  out->writeState(P_VAR, n, X);
  out->writeLogWeights(n, lws);
  out->writeNumberX(n, allNx);
  out->writeEvidence(n, evidence);
  out->writeEss(n, ess);
  out->writeAcceptanceRate(n, acceptanceRate);
}

#endif
