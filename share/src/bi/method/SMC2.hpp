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
#include "../state/ThetaParticle.hpp"
#include "../state/Schedule.hpp"
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
 * Sequential Monte Carlo squared (SMC^2).
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam F ParticleMarginalMetropolisHastings type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 Output type.
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
   * @param pmmh PMMH sampler.
   * @param resam Resampler for theta-particles.
   * @param Nmoves Number of steps per \f$\theta\f$-particle.
   * @param adapter Proposal adaptation strategy.
   * @param adapterScale Scaling factor for local proposals.
   * @param out Output.
   */
  SMC2(B& m, F* pmmh = NULL, R* resam = NULL, const int Nmoves = 1,
      const SMC2Adapter adapter = NO_ADAPTER, const real adapterScale = 0.5,
      const real adapterEssRel = 0.25, IO1* out = NULL);

  /**
   * @name High-level interface.
   */
  //@{
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
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s Prototype state. The set of \f$\theta\f$-particles are
   * constructed from this.
   * @param inInit Initialisation file.
   * @param C Number of \f$\theta\f$-particles.
   */
  template<Location L, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, ThetaParticle<B,L>& s, IO2* inInit = NULL,
      const int C = 1);
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
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param s Prototype state. The set of \f$\theta\f$-particles are
   * constructed from this.
   * @param[out] thetas The \f$\theta\f$-particles.
   * @param[out] lws Log-weights of \f$\theta\f$-particles.
   * @param[out] as Ancestors of \f$\theta\f$-particles.
   *
   * @return Evidence.
   */
  template<Location L, class V1, class V2>
  real init(Random& rng, const ScheduleElement now, ThetaParticle<B,L>& s,
      std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, V2 as);

  /**
   * Step \f$x\f$-particles forward.
   *
   * @tparam L Location.
   * @tparam V1 Vector type
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[out] s Working state.
   * @param[in,out] thetas The \f$\theta\f$-particles.
   * @param[in,out] lws Log-weights of \f$\theta\f$-particles.
   * @param[in,out] as Ancestors of \f$\theta\f$-particles.
   *
   * @return Evidence.
   */
  template<Location L, class V1, class V2>
  real step(Random& rng, const ScheduleIterator first, ScheduleIterator& iter,
      const ScheduleIterator last, ThetaParticle<B,L>& s,
      std::vector<ThetaParticle<B,L>*>& thetas, V1 lws, V2 as);

  /**
   * Adapt proposal distribution.
   *
   * @tparam L Location.
   * @tparam V1 Vector type
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param thetas The \f$\theta\f$-particles.
   * @param lws Log-weights of \f$\theta\f$-particles.
   * @param[out] q The proposal distribution.
   */
  template<Location L, class V1, class V2, class M2>
  void adapt(const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
      GaussianPdf<V2,M2>& q);

  /**
   * Resample \f$\theta\f$-particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] lws Log-weights of \f$\theta\f$-particles.
   * @param[out] as Ancestors of \f$\theta\f$-particles.
   */
  template<Location L, class V1, class V2>
  void resample(Random& rng, const ScheduleElement now, V1 lws, V2 as,
      std::vector<ThetaParticle<B,L>*>& thetas);

  /**
   * Rejuvenate \f$\theta\f$-particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param first Start of time schedule.
   * @param now Current position in time schedule.
   * @param[out] s Working state.
   * @param[in,out] thetas The set of \f$\theta\f$-particles.
   * @param q Proposal distribution.
   * @param ess Current effective sample size.
   *
   * @return Acceptance rate.
   */
  template<Location L, class V1, class M1>
  real rejuvenate(Random& rng, const ScheduleIterator first,
      const ScheduleIterator now, ThetaParticle<B,L>& s,
      const std::vector<ThetaParticle<B,L>*>& thetas, GaussianPdf<V1,M1>& q, const real ess);

  /**
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param thetas Theta-particles.
   * @param lws Log-weights of theta-particles.
   * @param les Incremental log-evidences.
   */
  template<bi::Location L, class V1, class V2>
  void output(const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
      const V2 les);

  /**
   * Report progress on stderr.
   *
   * @param now Current step in time schedule.
   * @param ess Effective sample size of theta-particles.
   * @param r Was resampling performed?
   * @param acceptRate Acceptance rate of rejuvenation step (if any).
   */
  void report(const ScheduleElement now, const real ess, const bool r,
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
   * Number of PMMH steps when rejuvenating.
   */
  int Nmoves;

  /**
   * Rejuvenate with local moves?
   */
  SMC2Adapter adapter;

  /**
   * Scale of local proposal standard deviation when rejuvenating, relative
   * to sample standard deviation.
   */
  real adapterScale;

  /**
   * Relative ESS trigger for adaptation.
   */
  real adapterEssRel;

  /**
   * Output.
   */
  IO1* out;

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
      const int Nmoves = 1, const SMC2Adapter adapter = NO_ADAPTER,
      const real adapterScale = 0.5, const real adapterEssRel = 0.25,
      IO1* out = NULL) {
    return new SMC2<B,F,R,IO1>(m, pmmh, resam, Nmoves, adapter, adapterScale,
        adapterEssRel, out);
  }
};
}

#include "../math/misc.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

#include "boost/typeof/typeof.hpp"

template<class B, class F, class R, class IO1>
bi::SMC2<B,F,R,IO1>::SMC2(B& m, F* pmmh, R* resam, const int Nmoves,
    const SMC2Adapter adapter, const real adapterScale, const real adapterEssRel, IO1* out) :
m(m), pmmh(pmmh), resam(resam), Nmoves(Nmoves), adapter(adapter), adapterScale(
    adapterScale), adapterEssRel(adapterEssRel), out(out) {
  //
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class IO2>
void bi::SMC2<B,F,R,IO1>::sample(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, ThetaParticle<B,L>& s, IO2* inInit,
    const int C) {
  typedef typename temp_host_vector<real>::type host_logweight_vector_type;
  typedef typename temp_host_vector<int>::type host_ancestor_vector_type;

  std::vector<ThetaParticle<B,L>*> thetas(C);  // theta-particles
  host_logweight_vector_type lws(C);  // log-weights of theta-particles
  host_logweight_vector_type les(last->indexOutput() - first->indexOutput());  // incremental log-evidences
  host_ancestor_vector_type as(C);  // ancestors of theta-particles
  real le = 0.0;
  ScheduleIterator iter = first;

  /* init */
  le = init(rng, *iter, s, thetas, lws, as);
  int k = iter->indexOutput();
  les(k) = le;
  while (iter + 1 != last) {
    le = step(rng, first, iter, last, s, thetas, lws, as);
    k = iter->indexOutput();
    les(k) = le;
  }
  output(thetas, lws, les);

  /* delete remaining stuff */
  for (int i = 0; i < C; i++) {
    delete thetas[i];
  }

  /* terminate */
  term();
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
real bi::SMC2<B,F,R,IO1>::init(Random& rng, const ScheduleElement now,
    ThetaParticle<B,L>& s, std::vector<ThetaParticle<B,L>*>& thetas, V1 lws,
    V2 as) {
  /* pre-condition */
  assert(!V1::on_device);
  assert(!V2::on_device);

  real le = 0.0;
  int i;

  /* initialise theta-particles */
  for (i = 0; i < thetas.size(); ++i) {
    thetas[i] = new ThetaParticle<B,L>(s.size(), s.getTrajectory().size2());
    BOOST_AUTO(&theta, *thetas[i]);
    BOOST_AUTO(filter, pmmh->getFilter());

    m.parameterSample(rng, theta);
    theta.get(PY_VAR) = theta.get(P_VAR);
    theta.getParameters1() = row(theta.get(P_VAR), 0);
    theta.getLogPrior1() = m.parameterLogDensity(theta);

    /* initialise x-particles for this theta */
    filter->setOutput(&theta.getOutput());
    filter->init(rng, theta.getParameters1(), now, theta,
        theta.getLogWeights(), theta.getAncestors());
    theta.getIncLogLikelihood() = filter->correct(now, theta,
        theta.getLogWeights());
    theta.getLogLikelihood1() = theta.getIncLogLikelihood();
    filter->output(now, theta, 0, theta.getLogWeights(),
        theta.getAncestors());

    /* initialise weights and ancestors */
    lws(i) = theta.getIncLogLikelihood();
    as(i) = i;
  }
  le = logsumexp_reduce(lws) - bi::log(static_cast<real>(thetas.size()));

  return le;
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
real bi::SMC2<B,F,R,IO1>::step(Random& rng, const ScheduleIterator first,
    ScheduleIterator& iter, const ScheduleIterator last,
    ThetaParticle<B,L>& s, std::vector<ThetaParticle<B,L>*>& thetas, V1 lws,
    V2 as) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  int i, r;
  real le = 0.0, acceptRate = 0.0, ess = 0.0;
  GaussianPdf<host_vector_type,host_matrix_type> q(NP);  // proposal distro

  ess = resam->ess(lws);
  r = iter->hasObs() && resam->isTriggered(lws);
  if (r) {
    /* resample-move */
    adapt(thetas, lws, q);
    resample(rng, *iter, lws, as, thetas);
    acceptRate = rejuvenate(rng, first, iter + 1, s, thetas, q, ess);
  } else {
    Resampler::normalise(lws);
  }
  report(*iter, ess, r, acceptRate);

  ScheduleIterator iter1;

  for (i = 0; i < thetas.size(); i++) {
    BOOST_AUTO(&theta, *thetas[i]);
    BOOST_AUTO(filter, pmmh->getFilter());
    iter1 = iter;

    filter->setOutput(&theta.getOutput());
    theta.getIncLogLikelihood() = filter->step(rng, iter1, last, theta,
        theta.getLogWeights(), theta.getAncestors());
    theta.getLogLikelihood1() += theta.getIncLogLikelihood();

    /* compute incremental evidence along the way */
    // evidence needs to be updated with the previous weights
    // otherwise we count the new incremental likelihood twice!!
    lws(i) += theta.getIncLogLikelihood();

    filter->sampleTrajectory(rng, theta.getTrajectory());

  }
  le = logsumexp_reduce(lws) - bi::log(static_cast<real>(lws.size()));
  iter = iter1;

  return le;
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2, class M2>
void bi::SMC2<B,F,R,IO1>::adapt(
    const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    GaussianPdf<V2,M2>& q) {
  typedef typename sim_temp_vector<V2>::type vector_type;
  typedef typename sim_temp_matrix<M2>::type matrix_type;

  if (adapter != NO_ADAPTER) {
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
    if (adapter == LOCAL_ADAPTER) {
      matrix_scal(adapterScale, U);
      q.mean().clear();
    } else {
      q.setMean(mu);
    }
    q.setStd(U);
    q.init();
  }
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::SMC2<B,F,R,IO1>::resample(Random& rng, const ScheduleElement now,
    V1 lws, V2 as, std::vector<ThetaParticle<B,L>*>& thetas) {
  if (now.hasObs()) {
    resam->resample(rng, lws, as, thetas);
  }
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class M1>
real bi::SMC2<B,F,R,IO1>::rejuvenate(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    ThetaParticle<B,L>& s, const std::vector<ThetaParticle<B,L>*>& thetas,
    GaussianPdf<V1,M1>& q, const real ess) {
  typedef typename temp_host_vector<real>::type host_vector_type;
  typedef typename temp_host_matrix<real>::type host_matrix_type;

  const int P = thetas.size();
  int p, move, naccept = 0;
  bool accept = false;

  for (p = 0; p < P; ++p) {
    BOOST_AUTO(&theta, *thetas[p]);
    s.resize(theta.size());
    s = theta;
    for (move = 0; move < Nmoves; ++move) {
      pmmh->getFilter()->setOutput(&s.getOutput());
      if (adapter == NO_ADAPTER || ess < P*adapterEssRel) {
        accept = pmmh->step(rng, first, last, s);
      } else {
        accept = pmmh->step(rng, first, last, s, q, adapter == LOCAL_ADAPTER);
      }
      if (accept) {
        ++naccept;

        theta.resize(s.size(), false);
        theta = s;  ///@todo Avoid full copy, especially of cache
      }
    }
  }
  int totalMoves = (Nmoves * p);
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
  boost::mpi::all_reduce(world, &totalMoves, 1, &totalMoves, std::plus<int>());
  boost::mpi::all_reduce(world, &naccept, 1, &naccept, std::plus<int>());
#endif
  return static_cast<double>(naccept) / totalMoves;
}

template<class B, class F, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::SMC2<B,F,R,IO1>::output(
    const std::vector<ThetaParticle<B,L>*>& thetas, const V1 lws,
    const V2 les) {
  if (out != NULL) {
    pmmh->setOutput(out);
    for (int p = 0; p < thetas.size(); ++p) {
      pmmh->output(p, *thetas[p]);
    }
    out->writeLogWeights(lws);
    out->writeLogEvidences(les);
  }
}

template<class B, class F, class R, class IO1>
void bi::SMC2<B,F,R,IO1>::report(const ScheduleElement now, const real ess,
    const bool r, const real acceptRate) {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
#else
  const int rank = 0;
#endif

  if (rank == 0) {
    std::cerr << now.indexOutput() << ":\ttime " << now.getTime() << "\tESS "
        << ess;
    if (r) {
      std::cerr << "\tresample-move with acceptance rate " << acceptRate;
    }
    std::cerr << std::endl;
  }
}

template<class B, class F, class R, class IO1>
void bi::SMC2<B,F,R,IO1>::term() {
  //
}

#endif
