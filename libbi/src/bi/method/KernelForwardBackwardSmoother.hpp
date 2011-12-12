/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally
 * indii/ml/filter/KernelForwardBackwardSmoother.hpp
 */
#ifndef BI_METHOD_KERNELFORWARDBACKWARDSMOOTHER_HPP
#define BI_METHOD_KERNELFORWARDBACKWARDSMOOTHER_HPP

#include "Simulator.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"
#include "../pdf/KernelDensityPdf.hpp"
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
   * Proposal types.
   */
  enum proposal_type {
    /**
     * Use filter density as proposal.
     */
    FILTER,

    /**
     * Use backward propagation as proposal.
     */
    BACKWARD
  };

  /**
   * Vector type.
   */
  typedef host_vector<real> vector_type;

  /**
   * Matrix type.
   */
  typedef host_matrix<real> matrix_type;

  /**
   * Kernel density type.
   */
  typedef KernelDensityPdf<vector_type,matrix_type,S1,K1> kernel_density_type;

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
   * Smooth output of particle filter.
   *
   * @tparam IO3 #concept::ParticleFilterBuffer type.
   *
   * @param theta Static state.
   * @param s State.
   * @param in Output of particle filter.
   * @param type Proposal type.
   */
  template<bi::Location L, class IO3>
  void smooth(Static<L>& theta, State<L>& s, IO3* in,
      const proposal_type type = BACKWARD);

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
   * @param[out] fX2 Uncorrected filter samples at end time.
   * @param[out] flw2 Uncorrected filter log-weights at end time.
   * @param[out] sX Smoothed samples at end time.
   * @param[out] slw Smoothed log-weights at end time.
   */
  template<Location L, class IO3, class M1, class V1>
  void init(Static<L>& theta, IO3* in, M1 fX2, V1 flw2, M1 sX, V1 slw);

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
   * Compute smooth weights, where proposal samples drawn from filter
   * density.
   *
   * @tparam Q1 Pdf type.
   * @tparam Q2 Pdf type.
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param f2 Uncorrected (predicted) filter density at time \f$t(n)\f$.
   * @param s Smooth density at time \f$t(n)\f$.
   * @param qX2 Proposal samples propagated to time \f$t(n)\f$.
   * @param[out] Smooth weights.
   */
  template<class Q1, class Q2, class M1, class V1>
  void correct(Q1& f2, Q2& s, const M1 qX2, V1 slw);

  /**
   * Compute smooth weights, where proposal samples drawn from some arbitrary
   * kernel density proposal.
   *
   * @tparam Q1 Pdf type.
   * @tparam Q2 Pdf type.
   * @tparam Q3 Pdf type.
   * @tparam Q4 Pdf type.
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param f1 Filter density at time \f$t(n - 1)\f$.
   * @param f2 Uncorrected (predicted) filter density at time \f$t(n)\f$.
   * @param s Smooth density at time \f$t(n)\f$.
   * @param q Proposal density at time \f$t(n - 1)\f$.
   * @param qX1 Proposal samples at time \f$t(n - 1)\f$.
   * @param qX2 Proposal samples propagated to time \f$t(n)\f$.
   * @param[out] Smooth weights.
   *
   */
  template<class Q1, class Q2, class Q3, class Q4, class M1, class V1>
  void correct(Q1& f1, Q2& f2, Q3& s, Q4& q1, const M1 qX1, const M1 qX2,
      V1 slw);

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
   * Smooth output of particle filter using filter proposal.
   *
   * @tparam IO3 #concept::ParticleFilterBuffer type.
   *
   * @param theta Static state.
   * @param s State.
   * @param in Output of particle filter.
   */
  template<bi::Location L, class IO3>
  void smoothUsingFilter(Static<L>& theta, State<L>& s, IO3* in);

  /**
   * Smooth output of particle filter using backward propagation proposal.
   *
   * @tparam IO3 #concept::ParticleFilterBuffer type.
   *
   * @param theta Static state.
   * @param s State.
   * @param in Output of particle filter.
   */
  template<bi::Location L, class IO3>
  void smoothUsingBackward(Static<L>& theta, State<L>& s, IO3* in);

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
template<bi::Location L, class IO3>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::smooth(
    Static<L>& theta, State<L>& s, IO3* in, const proposal_type proposalType) {
  switch (proposalType) {
  case FILTER:
    smoothUsingFilter(theta, s, in);
    break;
  case BACKWARD:
    smoothUsingBackward(theta, s, in);
    break;
  }
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class IO3>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::smoothUsingFilter(
    Static<L>& theta, State<L>& s, IO3* in) {
  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  typedef typename locatable_temp_matrix<L,real>::type M1;
  typedef typename locatable_temp_vector<L,real>::type V1;

  const int P = in->size1();
  const int T = in->size2();

  BOOST_AUTO(qX1, temp_matrix<M1>(P,M)); // proposal samples at t(n - 1)
  BOOST_AUTO(qX2, temp_matrix<M1>(P,M)); // proposal samples at t(n)
  BOOST_AUTO(fX1, temp_matrix<M1>(P,M)); // corrected filter samples at t(n - 1)
  BOOST_AUTO(fX2, temp_matrix<M1>(P,M)); // uncorrected (predicted) fiter samples at t(n)
  BOOST_AUTO(sX, temp_matrix<M1>(P,M));  // smooth samples at t(n)

  BOOST_AUTO(flw1, temp_vector<V1>(P)); // corrected filter log-weights at t(n - 1)
  BOOST_AUTO(flw2, temp_vector<V1>(P)); // uncorrected (predicted) filter log-weights at t(n)
  BOOST_AUTO(slw, temp_vector<V1>(P));  // smooth log-weights at t(n)
  BOOST_AUTO(qlw, temp_vector<V1>(P));  // proposal log-weights at t(n - 1)

  real tnxt;
  int n = T - 1, r;

  init(theta, in, *fX1, *flw1, *qX1, *qlw);
  output(n, *qX1, *qlw);
  while (n > 0) {
    /* update time */
    tnxt = state.t;
    std::cerr << tnxt << ' ';
    in->readTime(n - 1, state.t);

    /* smooth density at t(n) */
    sX->swap(*qX1);
    slw->swap(*qlw);
    kernel_density_type sp(*sX, *slw, K);

    /* uncorrected (predicted) filter density at t(n) */
    fX2->swap(*fX1);
    flw2->clear();
    in->readResample(n, r);
    assert(r); ///@todo Don't require resample at all times
    kernel_density_type f2(*fX2, *flw2, K);

    /* filter density at t(n - 1) */
    in->readState(D_NODE, n - 1, columns(*fX1, 0, ND));
    in->readState(C_NODE, n - 1, columns(*fX1, ND, NC));
    if (haveParameters) {
      in->readState(P_NODE, n - 1, columns(*fX1, ND + NC, NP));
    }
    in->readLogWeights(n - 1, *flw1);
    kernel_density_type f1(*fX1, *flw1, K);

    /* proposal samples at t(n - 1) */
    *qX1 = *fX1;
    f1.samples(rng, *qX1);

    /* propagate proposal samples forward */
    s.get(D_NODE) = columns(*qX1, 0, ND);
    s.get(C_NODE) = columns(*qX1, ND, NC);
    if (haveParameters) {
      theta.get(P_NODE) = columns(*qX1, ND + NC, NP);
    }
    predict(tnxt, theta, s);

    /* proposal samples at t(n) */
    columns(*qX2, 0, ND) = s.get(D_NODE);
    columns(*qX2, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*qX2, ND + NC, NP) = theta.get(P_NODE);
    }

    /* compute smoothed weights at t(n - 1) */
    correct(f2, sp, *qX2, *qlw);

    /* output */
    output(n - 1, *qX1, *qlw);
    --n;
  }
  synchronize();
  term(theta);
  std::cerr << std::endl;

  delete qX1;
  delete qX2;
  delete fX1;
  delete fX2;
  delete sX;
  delete flw1;
  delete flw2;
  delete slw;
  delete qlw;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class IO3>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::smoothUsingBackward(
    Static<L>& theta, State<L>& s, IO3* in) {
  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  typedef typename locatable_temp_matrix<L,real>::type M1;
  typedef typename locatable_temp_vector<L,real>::type V1;

  const int P = in->size1();
  const int T = in->size2();

  BOOST_AUTO(qX1, temp_matrix<M1>(P,M)); // proposal samples at t(n - 1)
  BOOST_AUTO(qX2, temp_matrix<M1>(P,M)); // proposal samples at t(n)
  BOOST_AUTO(fX1, temp_matrix<M1>(P,M)); // corrected filter samples at t(n - 1)
  BOOST_AUTO(fX2, temp_matrix<M1>(P,M)); // uncorrected (predicted) fiter samples at t(n)
  BOOST_AUTO(sX, temp_matrix<M1>(P,M));  // smooth samples at t(n)

  BOOST_AUTO(flw1, temp_vector<V1>(P)); // corrected filter log-weights at t(n - 1)
  BOOST_AUTO(flw2, temp_vector<V1>(P)); // uncorrected (predicted) filter log-weights at t(n)
  BOOST_AUTO(slw, temp_vector<V1>(P));  // smooth log-weights at t(n)
  BOOST_AUTO(qlw, temp_vector<V1>(P));  // proposal log-weights at t(n - 1)

  real t, tnxt;
  int n = T - 1, r;

  init(theta, in, *fX1, *flw1, *qX1, *qlw);
  output(n, *qX1, *qlw);
  while (n > 0) {
    /* update time */
    tnxt = state.t;
    std::cerr << tnxt << ' ';
    in->readTime(n - 1, t);

    /* smooth density at t(n) */
    sX->swap(*qX1);
    slw->swap(*qlw);
    kernel_density_type sp(*sX, *slw, K);

    /* uncorrected (predicted) filter density at t(n) */
    fX2->swap(*fX1);
    flw2->clear();
    in->readResample(n, r);
    assert(r); ///@todo Don't require resample at all times
    kernel_density_type f2(*fX2, *flw2, K);

    /* filter density at t(n - 1) */
    in->readState(D_NODE, n - 1, columns(*fX1, 0, ND));
    in->readState(C_NODE, n - 1, columns(*fX1, ND, NC));
    if (haveParameters) {
      in->readState(P_NODE, n - 1, columns(*fX1, ND + NC, NP));
    }
    in->readLogWeights(n - 1, *flw1);
    kernel_density_type f1(*fX1, *flw1, K);

    /* smooth samples at t(n) */
    //sp.samples(rng, *sX);
    //qlw->clear();
    *qlw = *slw;

    /* propagate smooth samples back */
    s.get(D_NODE) = columns(*sX, 0, ND);
    s.get(C_NODE) = columns(*sX, ND, NC);
    if (haveParameters) {
      theta.get(P_NODE) = columns(*sX, ND + NC, NP);
    }
    predict(t, theta, s);
    state.t = t;

    /* proposal density at t(n - 1) */
    columns(*qX1, 0, ND) = s.get(D_NODE);
    columns(*qX1, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*qX1, ND + NC, NP) = theta.get(P_NODE);
    }
    kernel_density_type q(*qX1, *qlw, K);

    /* proposal samples at t(n - 1) */
    q.samples(rng, *qX1);
    qlw->clear();

    /* propagate proposal samples forward */
    s.get(D_NODE) = columns(*qX1, 0, ND);
    s.get(C_NODE) = columns(*qX1, ND, NC);
    if (haveParameters) {
      theta.get(P_NODE) = columns(*qX1, ND + NC, NP);
    }
    predict(tnxt, theta, s);

    /* proposal samples at t(n) */
    columns(*qX2, 0, ND) = s.get(D_NODE);
    columns(*qX2, ND, NC) = s.get(C_NODE);
    if (haveParameters) {
      columns(*qX2, ND + NC, NP) = theta.get(P_NODE);
    }

    /* compute smoothed weights at t(n - 1) */
    correct(f1, f2, sp, q, *qX1, *qX2, *qlw);

    /* output */
    output(n - 1, *qX1, *qlw);
    --n;
  }
  synchronize();
  term(theta);
  std::cerr << std::endl;

  delete qX1;
  delete qX2;
  delete fX1;
  delete fX2;
  delete sX;
  delete flw1;
  delete flw2;
  delete slw;
  delete qlw;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<bi::Location L, class IO3, class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::init(
    Static<L>& theta, IO3* in, M1 fX2, V1 flw2, M1 sX, V1 slw) {
  /* pre-condition */
  assert (in != NULL);
  assert (fX2.size1() == sX.size1() && fX2.size2() == sX.size2());
  assert (flw2.size() == fX2.size1());
  assert (slw.size() == sX.size1());

  sim.init(theta);

  int r;
  int n = in->size2() - 1;
  in->readTime(n, state.t);
  in->readResample(n, r);
  in->readState(D_NODE, n, columns(sX, 0, ND));
  in->readState(C_NODE, n, columns(sX, ND, NC));
  if (SH == STATIC_OWN) {
    in->readState(P_NODE, n, columns(sX, ND + NC, NP));
  }
  fX2 = sX;
  in->readLogWeights(n, slw);
  flw2.clear();
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
template<class Q1, class Q2, class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::correct(
    Q1& f2, Q2& s, const M1 qX2, V1 slw) {
  /* pre-conditions */
  assert (qX2.size1() == slw.size());
  assert (qX2.size2() == f2.size());
  assert (qX2.size2() == s.size());

  BOOST_AUTO(lws, temp_vector<V1>(slw.size()));

  slw.clear();

  f2.logDensities(qX2, *lws);
  axpy(-1.0, *lws, slw);

  s.logDensities(qX2, *lws);
  axpy(1.0, *lws, slw);

  renormalise(slw);
  delete lws;
}

template<class B, class IO1, class IO2, class K1, class S1, bi::Location CL,
    bi::StaticHandling SH>
template<class Q1, class Q2, class Q3, class Q4, class M1, class V1>
void bi::KernelForwardBackwardSmoother<B,IO1,IO2,K1,S1,CL,SH>::correct(
    Q1& f1, Q2& f2, Q3& s, Q4& q, const M1 qX1, const M1 qX2, V1 slw) {
  /* pre-conditions */
  assert (qX1.size1() == slw.size());
  assert (qX2.size1() == slw.size());
  assert (qX1.size2() == qX2.size2());
  assert (qX1.size2() == f1.size());
  assert (qX2.size2() == f2.size());
  assert (qX2.size2() == s.size());
  assert (qX1.size2() == q.size());

  BOOST_AUTO(lws, temp_vector<V1>(slw.size()));

  //slw.clear();

  f1.logDensities(qX1, *lws);
  axpy(1.0, *lws, slw);

  f2.logDensities(qX2, *lws);
  axpy(-1.0, *lws, slw);

  s.logDensities(qX2, *lws);
  axpy(1.0, *lws, slw);

  q.logDensities(qX1, *lws);
  axpy(-1.0, *lws, slw);

  renormalise(slw);
  delete lws;
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
