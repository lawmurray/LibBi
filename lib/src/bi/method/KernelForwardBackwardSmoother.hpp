/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1406 $
 * $Date: 2011-04-15 11:44:22 +0800 (Fri, 15 Apr 2011) $
 *
 * Imported from dysii 1.4.0, originally
 * indii/ml/filter/KernelForwardBackwardSmoother.hpp
 */
#ifndef BI_METHOD_KERNELFORWARDBACKWARDSMOOTHER_HPP
#define BI_METHOD_KERNELFORWARDBACKWARDSMOOTHER_HPP

#include "Simulator.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"
#include "../updater/RUpdater.hpp"
#include "../kd/FastGaussianKernel.hpp"
#include "../kd/MedianPartitioner.hpp"

namespace bi {
/**
 * @internal
 *
 * State of KernelForwardBackwardSmoother.
 */
struct KernelForwardBackwardSmootherState {
  /**
   * Constructor.
   */
  KernelForwardBackwardSmootherState();

  /**
   * Current time.
   */
  real t;
};
}

bi::KernelForwardBackwardSmootherState::KernelForwardBackwardSmootherState() : t(0.0) {
  //
}

namespace bi {
/**
 * Kernel forward-backward smoother
 * @ref Murray2011b "Murray & Storkey (2011)".
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::ParticleSmootherBuffer type.
 * @tparam K1 Kernel type.
 * @tparam S1 Partitioner type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * KernelForwardBackwardSmoother is suitable for continuous time systems with
 * nonlinear transition and measurement functions, approximating state and
 * noise with indii::ml::aux::DiracMixturePdf distributions. It is
 * particularly suitable in situations where the transition density is
 * intractable, such as for transition functions defined using Stochastic
 * Differential Equations (SDEs).
 * 
 * A number of significant optimisations may be triggered using Flags. The
 * use of flags is entirely optional, and considered an advanced feature.
 * Not using flags will trigger the most generally applicable algorithms,
 * suitable in all situations. Using the right flags in the right situation
 * will give significant performance improvements. Using flags in the wrong
 * situation will give erroneous results. Be sure to understand the
 * assumptions implied by a flag, and be certain that those assumptions are
 * suitable, before putting it to use.
 */
template<class B, class IO1, class IO2, class K1 = FastGaussianKernel,
    class S1 = MedianPartitioner, Location CL = ON_HOST,
    StaticHandling SH = STATIC_SHARED>
class KernelForwardBackwardSmoother {
public:  
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param delta Time step for d- and r-nodes.
   * @param K Kernel.
   * @param in Forcings.
   * @param out Output.
   */
  KernelForwardBackwardSmoother(B& m, Random& rng, const K1& K,
      const S1& partitioner, const real delta = 1.0, IO1 *in = NULL,
      IO2 *out = NULL);

  /**
   * Destructor.
   */
  ~KernelForwardBackwardSmoother();

  /**
   * Get the current time.
   */
  real getTime();

  /**
   * @copydoc #concept::Filter::getOutput()
   */
  IO1* getOutput();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Smooth output of ParticleFilter.
   *
   * @tparam IO2 #concept::ParticleFilterBuffer type.
   *
   * @param theta Static state.
   * @param s State.
   * @param in Output of particle filter.
   * @param resam Resampler.
   */
  template<bi::Location L, class IO3, class R>
  void smooth(Static<L>& theta, State<L>& s, IO3* in, R* resam);

  /**
   * @copydoc #concept::Filter::reset()
   */
  void reset();
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
   * @tparam IO3 #concept::ParticleFilterBuffer type.
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param theta Static state.
   * @param in Output of particle filter.
   * @param[out] X3 Uncorrected filter samples at end time.
   * @param[out] lw3 Uncorrected filter log-weights at end time.
   * @param[out] X4 Smoothed samples at end time.
   * @param[out] lw4 Smoothed log-weights at end time.
   */
  template<Location L, class IO3, class M1, class V1>
  void init(Static<L>& theta, IO3* in, M1 X2, V1 lw2, M1 X3, V1 lw3);

  /**
   * Simulate importance samples forward.
   *
   * @tparam L Location.
   *
   * @param T Time to which to predict.
   * @param theta Static state.
   * @param[in,out] s State.
   */
  template<bi::Location L>
  void predict(const real T, Static<L>& theta, State<L>& s);

  /**
   * Correct to obtain smoothed weights.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param X3 Uncorrected filter samples at next time.
   * @param lw3 Uncorrected filter log-weights at next time.
   * @param X4 Smoothed samples at next time.
   * @param lw4 Smoothed log-weights at next time.
   * @param X2 Importance samples propagated to next time.
   * @param[in,out] lw1 Log-weights of importance samples on input, smoothed
   * weights at current time on output.
   */
  template<class M1, class V1>
  void correct(const M1 X3, const V1 lw3, const M1 X4, const V1 lw4,
      const M1 X2, V1 lw1);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param theta Static state.
   * @param s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   */
  template<Location L, class V1, class V2, class R>
  void resample(Static<L>& theta, State<L>& s, V1& lws, V2& as,
      R* resam = NULL);

  /**
   * Output.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param X Smooth samples.
   * @param lw Smooth log-weights.
   */
  template<class M1, class V1>
  void output(const int k, const M1 X, const V1 lw);

  /**
   * Clean up.
   */
  template<Location L>
  void term(Static<L>& theta);
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Random number generator.
   */
  Random& rng;

  /**
   * Size of state, excluding random variates and observations.
   */
  int M;

  /**
   * \f$\mathcal{K}(\|\mathbf{x}\|_p) \f$; the kernel.
   */
  K1 K;

  /**
   * Partitioner.
   */
  S1 partitioner;

  /**
   * R-net updater.
   */
  RUpdater<B> rUpdater;

  /**
   * Simulator.
   */
  Simulator<B,RUpdater<B>,IO1,SimulatorNetCDFBuffer,CL,SH> sim;

  /**
   * Log-variables.
   */
  std::set<int> logs;

  /**
   * Output.
   */
  IO2* out;

  /**
   * State.
   */
  KernelForwardBackwardSmootherState state;

  /**
   * Estimate parameters as well as state?
   */
  static const bool haveParameters = SH == STATIC_OWN;

  /**
   * Is out not null?
   */
  bool haveOut;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
};

/**
 * Factory for creating KernelForwardBackwardSmoother objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @see KernelForwardBackwardSmoother
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct KernelForwardBackwardSmootherFactory {
  /**
   * Create kernel forward-backward smoother.
   *
   * @return KernelForwardBackwardSmoother object. Caller has ownership.
   *
   * @see KernelForwardBackwardSmoother::KernelForwardBackwardSmoother()
   */
  template<class B, class IO1, class IO2, class K1, class S1>
  static KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>* create(B& m,
      Random& rng, K1& K, S1& partitioner, const real delta = 1.0,
      IO1* in = NULL, IO2* out = NULL) {
    return new KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>(m, rng,
        K, partitioner, delta, in, out);
  }
};
}

#include "../pdf/KernelDensityPdf.hpp"

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::KernelForwardBackwardSmoother(
    B& m, Random& rng, const K1& K, const S1& partitioner, const real delta,
    IO1* in, IO2* out) :
    m(m),
    rng(rng),
    M(ND + NC + (haveParameters ? NP : 0)),
    K(K),
    partitioner(partitioner),
    rUpdater(rng),
    sim(m, delta, &rUpdater, in), out(out),
    haveOut(out != NULL && out->size2() > 0) {
  /* set up log variables */
  offset_insert(logs, m.getLogs(D_NODE).begin(), m.getLogs(D_NODE).end());
  offset_insert(logs, m.getLogs(C_NODE).begin(), m.getLogs(C_NODE).end(), ND);
  if (haveParameters) {
    offset_insert(logs, m.getLogs(P_NODE).begin(), m.getLogs(P_NODE).end(), ND + NC);
  }
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::~KernelForwardBackwardSmoother() {
  //
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
inline real bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
inline IO1* bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class IO3, class R>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::smooth(
    Static<L>& theta, State<L>& s, IO3* in, R* resam) {
  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  typedef typename locatable_temp_matrix<L,real>::type M1;
  typedef typename locatable_temp_vector<L,real>::type V1;

  const int P = in->size1();
  const int T = in->size2();

  BOOST_AUTO(X0, temp_matrix<M1>(P,M));
  BOOST_AUTO(X1, temp_matrix<M1>(P,M));
  BOOST_AUTO(X2, temp_matrix<M1>(P,M));
  BOOST_AUTO(X3, temp_matrix<M1>(P,M));
  BOOST_AUTO(X4, temp_matrix<M1>(P,M));
  BOOST_AUTO(lw0, temp_vector<V1>(P));
  BOOST_AUTO(lw1, temp_vector<V1>(P));
  BOOST_AUTO(lw2, temp_vector<V1>(P));
  BOOST_AUTO(lw3, temp_vector<V1>(P));
  BOOST_AUTO(lw4, temp_vector<V1>(P));
  BOOST_AUTO(as, host_temp_vector<int>(P));

  real tnxt;
  int n = T - 1, r;
  init(theta, in, *X0, *lw0, *X1, *lw1);
  output(n, *X1, *lw1);

  while (n > 0) {
    tnxt = state.t;
    std::cerr << tnxt << ' ';

    /* input */
    in->readTime(n - 1, state.t);
    in->readState(D_NODE, n - 1, s.get(D_NODE));
    in->readState(C_NODE, n - 1, s.get(C_NODE));
    in->readState(R_NODE, n - 1, s.get(R_NODE));
    if (haveParameters) {
      in->readState(P_NODE, n - 1, theta.get(P_NODE));
    }
    in->readLogWeights(n, *lw0);

    /* smooth samples and weights at t(n) */
    X4->swap(*X1);
    lw4->swap(*lw1);

    /* uncorrected filter samples and weights at t(n) */
    /**
     * @todo Won't work with AuxiliaryParticleFilter, as after resampling,
     * weights not necessarily uniform.
     */
    X3->swap(*X0);
    in->readResample(n, r);
    if (r) {
      lw3->clear();
    } else {
      *lw3 = *lw0;
    }

    /* pre-resampling filter samples at t(n - 1) */
    columns(*X0, 0, ND) = s.get(D_NODE);
    columns(*X0, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*X0, ND + NC, NP) = theta.get(P_NODE);
    }

    /* resample */
    *lw1 = *lw0;
    resample(theta, s, *lw1, *as, resam);

    /* post-resampling filter samples at t(n - 1) */
    columns(*X1, 0, ND) = s.get(D_NODE);
    columns(*X1, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*X1, ND + NC, NP) = theta.get(P_NODE);
    }

    /* propagate */
    predict(tnxt, theta, s);

    /* post-propagation filter samples at t(n) */
    columns(*X2, 0, ND) = s.get(D_NODE);
    columns(*X2, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*X2, ND + NC, NP) = theta.get(P_NODE);
    }

    /* compute smoothed weights at t(n - 1) */
    correct(*X3, *lw3, *X4, *lw4, *X2, *lw1);

    /* output */
    output(n - 1, *X1, *lw1);
    --n;
  }
  synchronize();
  term(theta);
  std::cerr << std::endl;

  delete X0;
  delete X1;
  delete X2;
  delete X3;
  delete X4;
  delete lw0;
  delete lw1;
  delete lw2;
  delete lw3;
  delete lw4;
  delete as;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class IO3, class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::init(
    Static<L>& theta, IO3* in, M1 X3, V1 lw3, M1 X4, V1 lw4) {
  /* pre-condition */
  assert (in != NULL);
  assert (X3.size1() == X4.size1() && X3.size2() == X4.size2());
  assert (lw3.size() == X3.size1());
  assert (lw4.size() == X4.size1());

  sim.init(theta);

  int r;
  int n = in->size2() - 1;
  in->readTime(n, state.t);
  in->readResample(n, r);
  in->readState(D_NODE, n, columns(X4, 0, ND));
  in->readState(C_NODE, n, columns(X4, ND, NC));
  if (SH == STATIC_OWN) {
    in->readState(P_NODE, n, columns(X4, ND + NC, NP));
  }
  X3 = X4;
  in->readLogWeights(n, lw4);
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::predict(
    const real T, Static<L>& theta, State<L>& s) {
  sim.setTime(state.t, s);
  if (haveParameters) {
    sim.init(theta);
  }
  sim.advance(T, s);
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::correct(
    const M1 X3, const V1 lw3, const M1 X4, const V1 lw4, const M1 X2, V1 lw1) {
  /* pre-conditions */
  assert (lw1.size() == X2.size1());
  assert (lw3.size() == X3.size1());
  assert (lw4.size() == X4.size1());
  assert (X2.size1() == X3.size1() && X2.size2() == X3.size2());
  assert (X2.size1() == X4.size1() && X2.size2() == X4.size2());

  const int P = X2.size1();
  BOOST_AUTO(lp3, temp_vector<V1>(P));
  BOOST_AUTO(lp4, temp_vector<V1>(P));
  
  /* build kernel density estimates */
  KernelDensityPdf<host_vector<real>,host_matrix<real>,S1,K1> uncorrected(X3, lw3, K, logs);
  KernelDensityPdf<host_vector<real>,host_matrix<real>,S1,K1> smooth(X4, lw4, K, logs);
  
  /* perform kernel density evaluations */
  uncorrected.logDensities(X2, *lp3);
  smooth.logDensities(X2, *lp4);

  /* compute smoothed weights */
  axpy(1.0, *lp4, lw1);
  axpy(-1.0, *lp3, lw1);

  /* renormalise */
  thrust::replace_if(lw1.begin(), lw1.end(), is_not_finite_functor<real>(), std::log(0.0));
  real mx = *bi::max(lw1.begin(), lw1.end());
  thrust::transform(lw1.begin(), lw1.end(), lw1.begin(), subtract_constant_functor<real>(mx));

  synchronize();
  delete lp3;
  delete lp4;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class R>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::resample(
    Static<L>& theta, State<L>& s, V1& lws, V2& as, R* resam) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (theta.size() == 1 || theta.size() == lws.size());

  if (resam != NULL) {
    resam->resample(lws, as, theta, s);
  }
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::output(
    const int k, const M1 X, const V1 lw) {
  if (haveOut) {
    out->writeTime(k, state.t);
    out->writeState(D_NODE, k, columns(X, 0, ND));
    out->writeState(C_NODE, k, columns(X, ND, NC));
    if (haveParameters) {
      out->writeState(P_NODE, k, columns(X, ND + NC, NP));
    }
    out->writeLogWeights(k, lw);
  }
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::term(
    Static<L>& theta) {
  sim.term(theta);
}

#endif
