/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_RAUCHTUNGSTRIEBELSMOOTHER_HPP
#define BI_METHOD_RAUCHTUNGSTRIEBELSMOOTHER_HPP

#include "misc.hpp"
#include "../misc/location.hpp"
#include "../misc/Markable.hpp"

namespace bi {
/**
 * @internal
 *
 * State of RauchTungStriebelSmoother.
 */
struct RauchTungStriebelSmootherState {
  /**
   * Constructor.
   */
  RauchTungStriebelSmootherState();

  /**
   * Current time.
   */
  real t;

  /**
   * Current index into input file.
   */
  int k;
};
}

bi::RauchTungStriebelSmootherState::RauchTungStriebelSmootherState() : t(0.0), k(0) {
  //
}

namespace bi {
/**
 *  Rauch-Tung-Striebel Smoother.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::KalmanFiltererBuffer type.
 * @tparam IO2 #concept::KalmanSmootherBuffer type.
 * @tparam CL Cache location.
 *
 * Implementation based on @ref Sarkka2008 "Särkkä (2008)".
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class IO1, class IO2, Location CL>
class RauchTungStriebelSmoother : public Markable<RauchTungStriebelSmootherState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Input.
   * @param out Output.
   */
  RauchTungStriebelSmoother(B& m, IO1* in, IO2* out = NULL);

  /**
   * Destructor.
   */
  ~RauchTungStriebelSmoother();

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
   * Smooth.
   */
  template<Location L>
  void smooth(State<B,L>& s);

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
   * @param[out] s State.
   */
  template<Location L>
  void init(State<B,L>& s);

  /**
   * Correct using smooth state at next time.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void correct(State<B,L>& s);

  /**
   * Output.
   *
   * @tparam L Location.
   *
   * @param k Time index.
   * @param s State.
   */
  template<Location L>
  void output(const int k, const State<B,L>& s);

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
   * Input.
   */
  IO1* in;

  /**
   * Output.
   */
  IO2* out;

  /**
   * State.
   */
  RauchTungStriebelSmootherState state;

  /* net sizes, for convenience */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int M =  NR + ND;
};

/**
 * Factory for creating RauchTungStriebelSmoother objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see RauchTungStriebelSmoother
 */
template<Location CL = ON_HOST>
struct RauchTungStriebelSmootherFactory {
  /**
   * Create RauchTungStriebel smoother.
   *
   * @return RauchTungStriebelSmoother object. Caller has ownership.
   *
   * @see RauchTungStriebelSmoother::RauchTungStriebelSmoother()
   */
  template<class B, class IO1, class IO2>
  static RauchTungStriebelSmoother<B,IO1,IO2,CL>* create(B& m, IO1* in, IO2* out = NULL) {
    return new RauchTungStriebelSmoother<B,IO1,IO2,CL>(m, in, out);
  }
};

}

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/pi.hpp"

template<class B, class IO1, class IO2, bi::Location CL>
bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::RauchTungStriebelSmoother(B& m, IO1* in, IO2* out) :
    m(m),
    in(in),
    out(out) {
  /* pre-condition */
  BI_ASSERT(in != NULL);

  reset();
}

template<class B, class IO1, class IO2, bi::Location CL>
bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::~RauchTungStriebelSmoother() {
  //
}

template<class B, class IO1, class IO2, bi::Location CL>
inline real bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, bi::Location CL>
inline IO1* bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::smooth(State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(in != NULL);
  BI_ASSERT(in->size2() > 0);

  int k = in->size2()/2 - 1;
  init(s);
  output(k, s);
  --k;

  while (k >= 0) {
    correct(s);
    output(k, s);
    --k;
  }
  synchronize();
  term();
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::reset() {
  state.t = 0.0;
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::init(State<B,L>& s) {
  typename temp_host_matrix<real>::type S(M,M);

  state.k = in->size2() - 1;
  in->readTime(state.k, state.t);
  in->readState(D_VAR, state.k, s.get(D_VAR));
  in->readState(R_VAR, state.k, s.get(R_VAR));
  in->readStd(state.k, S);
  subrange(s.getStd(), 0, M, 0, M) = S;

  --state.k;
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::correct(State<B,L>& s) {
  typename temp_host_matrix<real>::type S(M,M);

  in->readTime(state.k, state.t);
  in->readState(D_VAR, state.k, s.get(D_VAR));
  in->readState(R_VAR, state.k, s.get(R_VAR));
  in->readStd(state.k, S);
  subrange(s.getStd(), 0, M, 0, M) = S;
  --state.k;

  in->readTime(state.k, state.t);
  in->readState(D_VAR, state.k, s.get(D_VAR));
  in->readState(R_VAR, state.k, s.get(R_VAR));
  in->readStd(state.k, S);
  subrange(s.getStd(), 0, M, 0, M) = S;
  --state.k;

  //marginalise(corrected, nextUncorrected, SigmaXX, nextSmooth, smooth);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::output(const int k, const State<B,L>& s) {
  if (out != NULL) {
    typename temp_host_matrix<real>::type S(M,M);
    S = subrange(s.getStd(), 0, M, 0, M);

    out->writeTime(k, state.t);
    out->writeState(D_VAR, k, s.get(D_VAR));
    out->writeState(R_VAR, k, s.get(R_VAR));
    out->writeStd(k, S);
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::term() {
  //
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::mark() {
  Markable<RauchTungStriebelSmootherState>::mark(state);
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::RauchTungStriebelSmoother<B,IO1,IO2,CL>::restore() {
  Markable<RauchTungStriebelSmootherState>::restore(state);
}

#endif
