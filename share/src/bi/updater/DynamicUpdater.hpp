/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_DYNAMICUPDATER_HPP
#define BI_UPDATER_DYNAMICUPDATER_HPP

namespace bi {
/**
 * Update using block.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicUpdater {
public:
  /**
   * Update state.
   *
   * @tparam T1 Scalar type.
   *
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   */
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);

  /**
   * Update single trajectory.
   *
   * @tparam T1 Scalar type.
   *
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p);

  #ifdef __CUDACC__
  /**
   * Update state.
   *
   * @tparam T1 Scalar type.
   *
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   */
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);

  /**
   * Update single trajectory.
   *
   * @tparam T1 Scalar type.
   *
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s,
      const int p);
  #endif
};
}

#include "../host/updater/DynamicUpdaterHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/updater/DynamicUpdaterSSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/updater/DynamicUpdaterGPU.cuh"
#endif

template<class B, class S>
template<class T1>
void bi::DynamicUpdater<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  #ifdef ENABLE_SSE
  if (s.size() % BI_SIMD_SIZE == 0) {
    DynamicUpdaterSSE<B,S>::update(t1, t2, s);
  } else {
    DynamicUpdaterHost<B,S>::update(t1, t2, s);
  }
  #else
  DynamicUpdaterHost<B,S>::update(t1, t2, s);
  #endif
}

template<class B, class S>
template<class T1>
void bi::DynamicUpdater<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, const int p) {
  DynamicUpdaterHost<B,S>::update(t1, t2, s, p);
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1>
void bi::DynamicUpdater<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  DynamicUpdaterGPU<B,S>::update(t1, t2, s);
}

template<class B, class S>
template<class T1>
void bi::DynamicUpdater<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s, const int p) {
  DynamicUpdaterGPU<B,S>::update(t1, t2, s, p);
}
#endif

#endif
