/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_UNSCENTEDRTSSMOOTHER_HPP
#define BI_METHOD_UNSCENTEDRTSSMOOTHER_HPP

#include "misc.hpp"
#include "../pdf/ExpGaussianPdf.hpp"
#include "../math/locatable.hpp"
#include "../misc/Markable.hpp"

namespace bi {
/**
 * @internal
 *
 * State of UnscentedRTSSmoother.
 */
struct UnscentedRTSSmootherState {
  /**
   * Constructor.
   */
  UnscentedRTSSmootherState();

  /**
   * Current time.
   */
  real t;
};
}

bi::UnscentedRTSSmootherState::UnscentedRTSSmootherState() : t(0.0) {
  //
}

namespace bi {
/**
 * Unscented Rauch-Tung-Striebel Smoother.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::UnscentedRTSSmootherBuffer type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * Implementation based on @ref Sarkka2008 "Särkkä (2008)".
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class IO1, Location CL, StaticHandling SH>
class UnscentedRTSSmoother : public Markable<UnscentedRTSSmootherState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param out Output.
   */
  UnscentedRTSSmoother(B& m, Random& rng, IO1* out = NULL);

  /**
   * Destructor.
   */
  ~UnscentedRTSSmoother();

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
   * Smooth output of UnscentedKalmanFilter.
   *
   * @tparam IO2 #concept::UnscentedKalmanFilterBuffer type.
   *
   * @param in Output of unscented Kalman filter.
   */
  template<class IO2>
  void smooth(IO2* in);

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
   * @tparam IO2 #concept::UnscentedKalmanFilterBuffer type.
   *
   * @param in Output of unscented Kalman filter.
   */
  template<class IO2, class V1, class M1>
  void init(IO2* in, ExpGaussianPdf<V1,M1>& smooth);

  /**
   * Correct using smooth state at next time.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param corrected Corrected state marginal at current time.
   * @param nextUncorrected Uncorrected state marginal at next time.
   * @param SigmaXX Uncorrected-corrected cross-covariance.
   * @param nextSmooth Smooth state marginal at next time.
   * @param[out] smooth Smooth state marginal at current time.
   */
  template<class V1, class M1>
  void correct(const ExpGaussianPdf<V1,M1>& corrected,
      const ExpGaussianPdf<V1,M1>& nextUncorrected, const M1& SigmaXX,
      const ExpGaussianPdf<V1,M1>& nextSmooth,
      ExpGaussianPdf<V1,M1>& smooth);

  /**
   * Output.
   *
   * @param k Time index.
   * @param smooth Smooth state marginal at current time.
   */
  template<class V1, class M1>
  void output(const int k, const ExpGaussianPdf<V1,M1>& smooth);

  /**
   * Clean up.
   */
  void term();
  //@}

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

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
   * Output.
   */
  IO1* out;

  /**
   * State.
   */
  UnscentedRTSSmootherState state;

  /**
   * Estimate parameters as well as state?
   */
  bool haveParameters;

  /**
   * Is out not null?
   */
  bool haveOut;

  /* net sizes, for convenience */
  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NO = net_size<B,typename B::OTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
};

/**
 * Factory for creating UnscentedRTSSmoother objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @see UnscentedRTSSmoother
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct UnscentedRTSSmootherFactory {
  /**
   * Create unscented RTS smoother.
   *
   * @return UnscentedRTSSmoother object. Caller has ownership.
   *
   * @see UnscentedRTSSmoother::UnscentedRTSSmoother()
   */
  template<class B, class IO1>
  static UnscentedRTSSmoother<B,IO1,CL,SH>* create(
      B& m, Random& rng, IO1* out = NULL) {
    return new UnscentedRTSSmoother<B,IO1,CL,SH>(m, rng, out);
  }
};

}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/pi.hpp"

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
bi::UnscentedRTSSmoother<B,IO1,CL,SH>::UnscentedRTSSmoother(B& m,
    Random& rng, IO1* out) :
    m(m),
    rng(rng),
    M(ND + NC + ((SH == STATIC_OWN) ? NP : 0)),
    out(out),
    haveParameters(SH == STATIC_OWN),
    haveOut(out != NULL) {
  reset();
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
bi::UnscentedRTSSmoother<B,IO1,CL,SH>::~UnscentedRTSSmoother() {
  //
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
inline real bi::UnscentedRTSSmoother<B,IO1,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
inline IO1* bi::UnscentedRTSSmoother<B,IO1,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<class IO2>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::smooth(IO2* in) {
  typedef host_vector<real> V1;
  typedef host_matrix<real> M1;

  /* pre-condition */
  assert (in != NULL);
  assert (in->size2() > 0);

  ExpGaussianPdf<V1,M1> corrected(M), nextUncorrected(M), nextSmooth(M), smooth(M);
  M1 SigmaXX(M,M), SigmaXX2(M,M);

  int n = in->size2() - 1;
  init(in, smooth);
  output(n, smooth);
  --n;

  while (n >= 0) {
    nextSmooth = smooth;
    in->readTime(n, state.t);
    in->readCorrectedState(n, corrected.mean(), corrected.cov());
    in->readUncorrectedState(n + 1, nextUncorrected.mean(), nextUncorrected.cov());
    in->readCrossState(n + 1, SigmaXX);
    transpose(SigmaXX, SigmaXX2);
    corrected.init();
    nextUncorrected.init();
    correct(corrected, nextUncorrected, SigmaXX, nextSmooth, smooth);
    output(n, smooth);
    --n;
  }
  synchronize();
  term();
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::reset() {
  state.t = 0.0;
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<class IO2, class V1, class M1>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::init(IO2* in,
    ExpGaussianPdf<V1,M1>& smooth) {
  /* pre-condition */
  assert (in != NULL);
  assert (smooth.size() == M);

  in->readTime(in->size2() - 1, state.t);
  in->readCorrectedState(in->size2() - 1, smooth.mean(), smooth.cov());
  smooth.init();
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<class V1, class M1>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::correct(
    const ExpGaussianPdf<V1,M1>& corrected,
    const ExpGaussianPdf<V1,M1>& nextUncorrected, const M1& SigmaXX,
    const ExpGaussianPdf<V1,M1>& nextSmooth,
    ExpGaussianPdf<V1,M1>& smooth) {
  marginalise(corrected, nextUncorrected, SigmaXX, nextSmooth, smooth);
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
template<class V1, class M1>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::output(const int k,
    const ExpGaussianPdf<V1,M1>& smooth) {
  if (haveOut) {
    out->writeTime(k, state.t);
    out->writeSmoothState(k, smooth.mean(), smooth.cov());
  }
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::term() {
  //
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::mark() {
  Markable<UnscentedRTSSmootherState>::mark(state);
}

template<class B, class IO1, bi::Location CL, bi::StaticHandling SH>
void bi::UnscentedRTSSmoother<B,IO1,CL,SH>::restore() {
  Markable<UnscentedRTSSmootherState>::restore(state);
}

#endif
