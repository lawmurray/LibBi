/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SIMULATOR_HPP
#define BI_METHOD_SIMULATOR_HPP

#include "../state/Static.hpp"
#include "../state/State.hpp"
#include "../updater/SUpdater.hpp"
#include "../updater/DUpdater.hpp"
#include "../updater/CUpdater.hpp"
#include "../updater/RUpdater.hpp"
#include "../updater/FUpdater.hpp"
#include "../updater/PUpdater.hpp"
#include "../buffer/SimulatorCache.hpp"
#include "../buffer/Cache1D.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SparseInputNetCDFBuffer.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * %State of Simulator.
 */
struct SimulatorState {
  /**
   * Constructor.
   */
  SimulatorState();

  /**
   * Time.
   */
  real t;
};
}

bi::SimulatorState::SimulatorState() : t(0.0) {
  //
}

namespace bi {
/**
 * %Simulator for dynamic models.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam U1 Updater type (for r-net).
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SimulatorBuffer type.
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class U1 = RUpdater<B>, class IO1 = SparseInputNetCDFBuffer,
    class IO2 = SimulatorNetCDFBuffer, Location CL = ON_HOST,
    StaticHandling SH = STATIC_SHARED>
class Simulator : public Markable<SimulatorState> {
public:
  /**
   * Type of caches.
   */
  typedef SimulatorCache<ON_HOST> cache_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param delta Time step for d- and r-nodes.
   * @param rUpdater Updater for r-net.
   * @param in Input for f-net.
   * @param out Output.
   */
  Simulator(B& m, const real delta = 1.0, U1* rUpdater = NULL,
      IO1* in = NULL, IO2* out = NULL);

  /**
   * Destructor.
   */
  ~Simulator();

  /**
   * Get output buffer.
   */
  IO2* getOutput();

  /**
   * Get cache.
   *
   * @param type Node type.
   *
   * @return Cache for given node type.
   */
  const cache_type& getCache(const NodeType type) const;

  /**
   * Get the current time.
   */
  real getTime();

  /**
   * Set the current time.
   *
   * @param t The time.
   * @param[out] s State.
   */
  template<Location L>
  void setTime(const real t, State<L>& s);

  /**
   * Reset to starting point.
   */
  void reset();

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Simulate forward.
   *
   * @tparam L Location.
   *
   * @param T Time to which to simulate.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   *
   * If an output buffer is given, it is filled with:
   *
   * @li the starting state, if it can hold two or more states,
   * @li the ending state, and
   * @li as many equispaced intermediate results as can be fit in between.
   *
   * Note, then, that if the buffer has space for storing only one state,
   * it is the ending state that is output.
   */
  template<Location L>
  void simulate(const real T, Static<L>& theta, State<L>& s);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam L Location.
   *
   * @param theta Static state.
   */
  template<Location L>
  void init(Static<L>& theta);

  /**
   * Simulate forward.
   *
   * @tparam L Location.
   *
   * @param tnxt Time to which to advance.
   * @param[in,out] s State.
   */
  template<bi::Location L>
  void advance(const real tnxt, State<L>& s);

  /**
   * Output static state.
   *
   * @tparam L Location.
   *
   * @param theta Static state.
   */
  template<bi::Location L>
  void output0(const Static<L>& theta);

  /**
   * Output state.
   *
   * @tparam L Location.
   *
   * @param k Time index.
   * @param s State.
   */
  template<bi::Location L>
  void output(const int k, const Static<L>& theta, const State<L>& s);

  /**
   * Flush state cache to file.
   *
   * @param type Node type.
   *
   * Flushes the cache for the given node type to file.
   */
  void flush(const NodeType type);

  /**
   * Flush all caches to file.
   */
  void flush();

  /**
   * Clean up.
   *
   * @tparam L Location.
   *
   * @param theta Static state.
   */
  template<bi::Location L>
  void term(Static<L>& theta);

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
   * Set cache mode.
   *
   * @param mode Mode.
   *
   * Sets the mode of all caches. Where this involves changing modes,
   * the existing cache is written out to the output buffer.
   */
  void setCacheMode(const typename cache_type::Mode mode);

  /**
   * Model.
   */
  B& m;

  /**
   * Time step.
   */
  real delta;

  /**
   * Updater for s-net.
   */
  SUpdater<B,SH> sUpdater;

  /**
   * Updater for d-net.
   */
  DUpdater<B,SH> dUpdater;

  /**
   * Updater for c-net.
   */
  CUpdater<B,SH> cUpdater;

  /**
   * Updater for f-net;
   */
  FUpdater<B,IO1,CL>* fUpdater;

  /**
   * Updater for p-net.
   */
  PUpdater<B> pUpdater;

  /**
   * Updater for r-net.
   */
  U1* rUpdater;

  /**
   * Output buffer.
   */
  IO2* out;

  /**
   * Cache for times.
   */
  Cache1D<real> timeCache;

  /**
   * Output caches.
   */
  std::vector<cache_type> caches;

  /**
   * Is rUpdater not null?
   */
  bool haveRUpdater;

  /**
   * Is fUpdater not null?
   */
  bool haveFUpdater;

  /**
   * Is out not null?
   */
  bool haveOut;

  /**
   * State.
   */
  SimulatorState state;
};

/**
 * Factory for creating Simulator objects.
 *
 * @ingroup method
 *
 * @see Simulator
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct SimulatorFactory {
  /**
   * Create simulator.
   *
   * @return Simulator object. Caller has ownership.
   *
   * @see Simulator::Simulator()
   */
  template<class B, class U1, class IO1, class IO2>
  static Simulator<B,U1,IO1,IO2,CL,SH>* create(B& m, const real delta = 1.0,
      U1* rUpdater = NULL, IO1* in = NULL, IO2* out = NULL) {
    return new Simulator<B,U1,IO1,IO2,CL,SH>(m, delta, rUpdater, in, out);
  }
};

}

#include "misc.hpp"

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
bi::Simulator<B,U1,IO1,IO2,CL,SH>::Simulator(B& m, const real delta,
    U1* rUpdater, IO1* in, IO2* out) :
    m(m), delta(delta), rUpdater(rUpdater), out(out), caches(NUM_NODE_TYPES),
    haveRUpdater(rUpdater != NULL), haveOut(out != NULL && out->size2() > 0) {
  haveFUpdater = in != NULL;
  fUpdater = (haveFUpdater) ? new FUpdater<B,IO1,CL>(*in) : NULL;
  reset();

  /* post-conditions */
  assert (rUpdater != NULL || !haveRUpdater);
  assert (fUpdater != NULL || !haveFUpdater);
  assert (!(out == NULL || out->size2() == 0) || !haveOut);
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
bi::Simulator<B,U1,IO1,IO2,CL,SH>::~Simulator() {
  flush();
  delete fUpdater;
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
inline IO2* bi::Simulator<B,U1,IO1,IO2,CL,SH>::getOutput() {
  return out;
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
inline const typename bi::Simulator<B,U1,IO1,IO2,CL,SH>::cache_type& bi::Simulator<B,U1,IO1,IO2,CL,SH>::getCache(
    const NodeType type) const {
  return caches[type];
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
inline real bi::Simulator<B,U1,IO1,IO2,CL,SH>::getTime() {
  return state.t;
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
inline void bi::Simulator<B,U1,IO1,IO2,CL,SH>::setTime(const real t, State<L>& s) {
  state.t = t;
  if (haveFUpdater) {
    fUpdater->setTime(t, s);
  }
}
template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
inline void bi::Simulator<B,U1,IO1,IO2,CL,SH>::reset() {
  Markable<SimulatorState>::unmark();
  state.t = 0.0;
  if (haveFUpdater) {
    fUpdater->reset();
  }
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::simulate(const real tnxt, Static<L>& theta, State<L>& s) {
  const real t0 = state.t;
  const int K = (haveOut) ? out->size2() : 1;
  real tk;
  int k;

  init(theta);
  output0(theta);
  if (K > 1) {
    output(0, theta, s);
  }
  for (k = 0; k == 0 || k < K - 1; ++k) { // enters at least once
    /* time of next output */
    tk = (k == K - 1) ? tnxt : t0 + (tnxt - t0)*(k + 1)/(K - 1);

    /* advance */
    advance(tk, s);

    /* output */
    if (K > 1) {
      output(k + 1, theta, s);
    } else {
      output(0, theta, s);
    }
  }
  synchronize();
  term(theta);
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::init(Static<L>& theta) {
  pUpdater.update(theta);
  sUpdater.update(theta);
  bind(theta);
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::advance(const real tnxt, State<L>& s) {
  /* pre-condition */
  assert (tnxt > state.t);

  real ti = state.t, tj, tf, td;
  if (haveFUpdater && fUpdater->hasNext() && fUpdater->getTime() >= ti) {
    tf = fUpdater->getTime();
  } else {
    tf = tnxt + 1.0;
  }
  if (m.getNumNodes(D_NODE) > 0 || (m.getNumNodes(R_NODE) && haveRUpdater)) {
    td = next_step(ti, delta);
  } else {
    td = tnxt + 1.0;
  }

  do { // over intermediate stopping points
    tj = std::min(tf, std::min(td, tnxt));

    if (ti >= tf) {
      /* update f-net */
      fUpdater->update(s);
      if (fUpdater->hasNext() && fUpdater->getTime() > tf) {
        tf = fUpdater->getTime();
      } else {
        tf = tnxt + 1.0;
      }
    }

    if (ti >= td) {
      td = next_step(td + 0.5*delta, delta);

      /* update r-net */
      if (haveRUpdater) {
        rUpdater->update(ti, td, s);
      }

      /* update d-net */
      dUpdater.update(ti, td, s);
    }

    /* update c-net */
    cUpdater.update(ti, tj, s);

    ti = tj;
  } while (ti < tnxt);
  state.t = tnxt;

  /* post-condition */
  assert (state.t == tnxt);
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::output0(const Static<L>& theta) {
  if (haveOut && SH == STATIC_SHARED) {
    setCacheMode(cache_type::STATE_MODE);
    caches[P_NODE].writeState(0, theta.get(P_NODE));
    caches[S_NODE].writeState(0, theta.get(S_NODE));
  }
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::output(const int k,
    const Static<L>& theta, const State<L>& s) {
  if (haveOut) {
    timeCache.put(k, state.t);
    setCacheMode(cache_type::STATE_MODE);
    caches[D_NODE].writeState(k, s.get(D_NODE));
    caches[C_NODE].writeState(k, s.get(C_NODE));
    caches[R_NODE].writeState(k, s.get(R_NODE));
    if (SH == STATIC_OWN) {
      caches[P_NODE].writeState(k, theta.get(P_NODE));
      caches[S_NODE].writeState(k, theta.get(S_NODE));
    }
  }
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::flush(const NodeType type) {
  int k, p, P;
  cache_type& cache = caches[type];

  synchronize();
  if (cache.getMode() == cache_type::STATE_MODE) {
    for (k = 0; k < cache.size2(); ++k) {
      if (cache.isValidState(k) && cache.isDirtyState(k)) {
        out->writeState(type, k, cache.getState(k));
      }
    }
  } else if (cache.getMode() == cache_type::TRAJECTORY_MODE) {
    for (k = 0; k < cache.size2(); ++k) {
      P = 0;
      p = 0;
      do {
        while (cache.isValidState(p + P) && cache.isDirtyState(p + P)) {
          ++P;
        }
        if (P > 0) {
          out->writeState(type, k, rows(cache.getState(k), p, P), p);
          p += P;
          P = 0;
        } else {
          ++p;
        }
      } while (p < cache.size1());
    }
  }
  cache.clean();
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::flush() {
  /* state caches */
  flush(D_NODE);
  flush(C_NODE);
  flush(R_NODE);
  if (SH == STATIC_OWN) {
    flush(P_NODE);
    flush(S_NODE);
  }

  /* time cache */
  int t = 0, T = 0;
  do {
    while (timeCache.isValid(t + T) && timeCache.isDirty(t + T)) {
      ++T;
    }
    if (T > 0) {
      out->writeTimes(t, T, subrange(timeCache.getPages(), t, T));
      t += T;
      T = 0;
    } else {
      ++t;
    }
  } while (t < timeCache.size());
  timeCache.clean();
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::term(Static<L>& theta) {
  unbind(theta);
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::setCacheMode(
    const typename cache_type::Mode mode) {
  int i;
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    if (caches[i].getMode() != mode) {
      flush((NodeType)i);
    }
    caches[i].setMode(mode);
  }
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::mark() {
  Markable<SimulatorState>::mark(state);
  if (haveFUpdater) {
    fUpdater->mark();
  }
}

template<class B, class U1, class IO1, class IO2, bi::Location CL, bi::StaticHandling SH>
void bi::Simulator<B,U1,IO1,IO2,CL,SH>::restore() {
  Markable<SimulatorState>::restore(state);
  if (haveFUpdater) {
    fUpdater->restore();
  }
}

#endif
