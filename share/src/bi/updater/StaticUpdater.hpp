/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_STATICUPDATER_HPP
#define BI_UPDATER_STATICUPDATER_HPP

namespace bi {
/**
 * Static updater.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticUpdater {
public:
  /**
   * Update state.
   *
   * @param[in,out] s State.
   */
  static void update(State<B,ON_HOST>& s);

  /**
   * Update single trajectory.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  static void update(State<B,ON_HOST>& s, const int p);

  #ifdef __CUDACC__
  /**
   * Update state.
   *
   * @param[in,out] s State.
   */
  static void update(State<B,ON_DEVICE>& s);

  /**
   * Update single trajectory.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  static void update(State<B,ON_DEVICE>& s, const int p);
  #endif
};
}

#include "../host/updater/StaticUpdaterHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/updater/StaticUpdaterSSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/updater/StaticUpdaterGPU.cuh"
#endif

template<class B, class S>
void bi::StaticUpdater<B,S>::update(State<B,ON_HOST>& s) {
  #ifdef ENABLE_SSE
  if (s.size() % BI_SIMD_SIZE == 0) {
    StaticUpdaterSSE<B,S>::update(s);
  } else {
    StaticUpdaterHost<B,S>::update(s);
  }
  #else
  StaticUpdaterHost<B,S>::update(s);
  #endif
}

template<class B, class S>
void bi::StaticUpdater<B,S>::update(State<B,ON_HOST>& s, const int p) {
  StaticUpdaterHost<B,S>::update(s, p);
}

#ifdef __CUDACC__
template<class B, class S>
void bi::StaticUpdater<B,S>::update(State<B,ON_DEVICE>& s) {
  StaticUpdaterGPU<B,S>::update(s);
}

template<class B, class S>
void bi::StaticUpdater<B,S>::update(State<B,ON_DEVICE>& s, const int p) {
  StaticUpdaterGPU<B,S>::update(s, p);
}
#endif

#endif
