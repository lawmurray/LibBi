/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1406 $
 * $Date: 2011-04-15 11:44:22 +0800 (Fri, 15 Apr 2011) $
 */
#ifndef BI_METHOD_PARTICLEFILTERSMOOTHER_HPP
#define BI_METHOD_PARTICLEFILTERSMOOTHER_HPP

#include "../buffer/ParticleSmootherNetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../state/Static.hpp"

namespace bi {
/**
 * @internal
 *
 * State of ParticleFilterSmoother.
 */
struct ParticleFilterSmootherState {
  /**
   * Constructor.
   */
  ParticleFilterSmootherState();

  /**
   * Current time.
   */
  real t;
};
}

bi::ParticleFilterSmootherState::ParticleFilterSmootherState() : t(0.0) {
  //
}

namespace bi {
/**
 * Particle filter-smoother.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::ParticleSmootherBuffer type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 */
template<class B, class IO1 = ParticleSmootherNetCDFBuffer,
    Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
class ParticleFilterSmoother {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param out Output.
   */
  ParticleFilterSmoother(B& m, IO1* out = NULL);

  /**
   * Destructor.
   */
  ~ParticleFilterSmoother();

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
   * @tparam L Location.
   * @tparam IO2 #concept::ParticleFilterBuffer type.
   *
   * @param theta Static state.
   * @param s State.
   * @param in Output of particle filter.
   */
  template<bi::Location L, class IO2>
  void smooth(Static<L>& theta, State<L>& s, IO2* in);

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
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   * @tparam IO2 #concept::ParticleFilterBuffer type.
   *
   * @param[out] theta Static state.
   * @param[out] s State.
   * @param[out] lw Log-weights.
   * @param[out] bs Ancestors.
   * @param in Output of particle filter.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(Static<L>& theta, State<L>& s, V1 lws, V2 bs, IO2* in);

  /**
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param theta Static state.
   * @param s State.
   * @param lw Smooth log-weights.
   */
  template<Location L, class V1>
  void output(const int k, const Static<L>& theta, const State<L>& s,
      const V1 lws);

  /**
   * Clean up.
   */
  void term();
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
   * State.
   */
  ParticleFilterSmootherState state;

  /**
   * Estimate parameters as well as state?
   */
  static const bool haveParameters = SH == STATIC_OWN;

  /**
   * Is out not null?
   */
  bool haveOut;
};

/**
 * Factory for creating ParticleFilterSmoother objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @see ParticleFilterSmoother
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct ParticleFilterSmootherFactory {
  /**
   * Create kernel forward-backward smoother.
   *
   * @return ParticleFilterSmoother object. Caller has ownership.
   *
   * @see ParticleFilterSmoother::ParticleFilterSmoother()
   */
  template<class B, class IO1>
  static ParticleFilterSmoother<B,IO1,CL,SH>* create(B& m, IO1* out = NULL) {
    return new ParticleFilterSmoother<B,IO1,CL,SH>(m, out);
  }
};
}

#include "Resampler.hpp"

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
bi::ParticleFilterSmoother<B,IO1,CL,SH>::ParticleFilterSmoother(
    B& m, IO1* out) :
    m(m),
    out(out),
    haveOut(out != NULL && out->size2() > 0) {
  //
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
bi::ParticleFilterSmoother<B,IO1,CL,SH>::~ParticleFilterSmoother() {
  //
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
inline real bi::ParticleFilterSmoother<B,IO1,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
inline IO1* bi::ParticleFilterSmoother<B,IO1,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class IO2>
void bi::ParticleFilterSmoother<B,IO1,CL,SH>::smooth(Static<L>& theta,
    State<L>& s, IO2* in) {
  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  const int P = in->size1();
  const int T = in->size2();

  BOOST_AUTO(lws, host_temp_vector<real>(P));
  BOOST_AUTO(as, host_temp_vector<int>(P));
  BOOST_AUTO(bs, host_temp_vector<int>(P));

  int n = T - 1;
  init(theta, s, *lws, *bs, in);
  output(n, theta, s, *lws);
  --n;
  while (n >= 0) {
    /* input */
    in->readTime(n, state.t);
    in->readState(D_NODE, n, s.get(D_NODE));
    in->readState(C_NODE, n, s.get(C_NODE));
    in->readState(R_NODE, n, s.get(R_NODE));
    if (haveParameters) {
      in->readState(P_NODE, n, theta.get(P_NODE));
    }
    in->readAncestors(n + 1, *as);

    /* trace back ancestry one step */
    thrust::gather(bs->begin(), bs->end(), as->begin(), bs->begin());

    /* resample */
    Resampler::copy(*bs, theta, s);

    /* output */
    output(n, theta, s, *lws);
    --n;
  }
  synchronize();
  term();

  delete lws;
  delete as;
  delete bs;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class IO2>
void bi::ParticleFilterSmoother<B,IO1,CL,SH>::init(
    Static<L>& theta, State<L>& s, V1 lws, V2 bs, IO2* in) {
  /* pre-condition */
  assert (in != NULL);

  int n = in->size2() - 1;
  in->readTime(n, state.t);
  in->readState(D_NODE, n, s.get(D_NODE));
  in->readState(C_NODE, n, s.get(C_NODE));
  in->readState(R_NODE, n, s.get(R_NODE));
  if (haveParameters) {
    in->readState(P_NODE, n, theta.get(P_NODE));
  }
  in->readLogWeights(n, lws);

  bi::sequence(bs.begin(), bs.end(), 0);
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::ParticleFilterSmoother<B,IO1,CL,SH>::output(const int k,
    const Static<L>& theta, const State<L>& s, const V1 lws) {
  if (haveOut) {
    out->writeTime(k, state.t);
    out->writeState(D_NODE, k, s.get(D_NODE));
    out->writeState(C_NODE, k, s.get(C_NODE));
    out->writeState(R_NODE, k, s.get(R_NODE));
    if (haveParameters) {
      out->writeState(P_NODE, k, theta.get(P_NODE));
    }
    out->writeLogWeights(k, lws);
  }
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
void bi::ParticleFilterSmoother<B,IO1,CL,SH>::term() {
  //
}

#endif
