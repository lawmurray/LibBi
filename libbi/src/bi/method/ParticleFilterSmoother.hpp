/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_PARTICLEFILTERSMOOTHER_HPP
#define BI_METHOD_PARTICLEFILTERSMOOTHER_HPP

#include "../buffer/ParticleSmootherNetCDFBuffer.hpp"
#include "../state/State.hpp"

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
 */
template<class B, class IO1 = ParticleSmootherNetCDFBuffer,
    Location CL = ON_HOST>
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
   * @param s State.
   * @param in Output of particle filter.
   */
  template<bi::Location L, class IO2>
  void smooth(State<B,L>& s, IO2* in);

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
   * @param[out] s State.
   * @param[out] lw Log-weights.
   * @param[out] bs Ancestors.
   * @param in Output of particle filter.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(State<B,L>& s, V1 lws, V2 bs, IO2* in);

  /**
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param s State.
   * @param lw Smooth log-weights.
   */
  template<Location L, class V1>
  void output(const int k, const State<B,L>& s, const V1 lws);

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
};

/**
 * Factory for creating ParticleFilterSmoother objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ParticleFilterSmoother
 */
template<Location CL = ON_HOST>
struct ParticleFilterSmootherFactory {
  /**
   * Create kernel forward-backward smoother.
   *
   * @return ParticleFilterSmoother object. Caller has ownership.
   *
   * @see ParticleFilterSmoother::ParticleFilterSmoother()
   */
  template<class B, class IO1>
  static ParticleFilterSmoother<B,IO1,CL>* create(B& m, IO1* out = NULL) {
    return new ParticleFilterSmoother<B,IO1,CL>(m, out);
  }
};
}

#include "Resampler.hpp"

template<class B, class IO1, bi::Location CL>
bi::ParticleFilterSmoother<B,IO1,CL>::ParticleFilterSmoother(
    B& m, IO1* out) :
    m(m),
    out(out) {
  //
}

template<class B, class IO1, bi::Location CL>
bi::ParticleFilterSmoother<B,IO1,CL>::~ParticleFilterSmoother() {
  //
}

template<class B, class IO1, bi::Location CL>
inline real bi::ParticleFilterSmoother<B,IO1,CL>::getTime() {
  return state.t;
}

template<class B, class IO1, bi::Location CL>
inline IO1* bi::ParticleFilterSmoother<B,IO1,CL>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class IO2>
void bi::ParticleFilterSmoother<B,IO1,CL>::smooth(
    State<B,L>& s, IO2* in) {
  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  const int P = in->size1();
  const int T = in->size2();

  temp_host_vector<real>::type lws(P);
  temp_host_vector<int>::type as(P), bs(P);

  int n = T - 1;
  init(theta, s, lws, bs, in);
  output(n, s, lws);
  --n;
  while (n >= 0) {
    /* input */
    in->readTime(n, state.t);
    in->readState(D_VAR, n, s.get(D_VAR));
    in->readState(R_VAR, n, s.get(R_VAR));
    in->readAncestors(n + 1, as);

    /* trace back ancestry one step */
    thrust::gather(bs.begin(), bs.end(), as.begin(), bs.begin());

    /* resample */
    Resampler::copy(bs, s);

    /* output */
    output(n, s, lws);
    --n;
  }
  synchronize();
  term();
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class V1, class V2, class IO2>
void bi::ParticleFilterSmoother<B,IO1,CL>::init(
    State<B,L>& s, V1 lws, V2 bs, IO2* in) {
  /* pre-condition */
  assert (in != NULL);

  int n = in->size2() - 1;
  in->readTime(n, state.t);
  in->readState(D_VAR, n, s.get(D_VAR));
  in->readState(R_VAR, n, s.get(R_VAR));
  in->readLogWeights(n, lws);

  seq_elements(bs, 0);
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class V1>
void bi::ParticleFilterSmoother<B,IO1,CL>::output(const int k,
    const State<B,L>& s, const V1 lws) {
  if (out != NULL) {
    out->writeTime(k, state.t);
    out->writeState(D_VAR, k, s.get(D_VAR));
    out->writeState(R_VAR, k, s.get(R_VAR));
    out->writeLogWeights(k, lws);
  }
}

template<class B, class IO1, bi::Location CL>
void bi::ParticleFilterSmoother<B,IO1,CL>::term() {
  //
}

#endif
