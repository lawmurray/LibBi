/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SPARSESTATICUPDATER_HPP
#define BI_UPDATER_SPARSESTATICUPDATER_HPP

namespace bi {
/**
 * Sparse static updater.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticUpdater {
public:
  /**
   * Update state.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   */
  static void update(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask);

  /**
   * Update single trajectory.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   * @param p Trajectory index.
   */
  static void update(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      const int p);

  #ifdef ENABLE_CUDA
  /**
   * Update state.
   *
   *
   * @param[in,out] s State.
   * @param mask Mask.
   */
  static void update(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask);

  /**
   * Update single trajectory.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   * @param p Trajectory index.
   */
  static void update(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
      const int p);
  #endif
};
}

#include "../host/updater/SparseStaticUpdaterHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/SparseStaticUpdaterGPU.cuh"
#endif

template<class B, class S>
void bi::SparseStaticUpdater<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask) {
  SparseStaticUpdaterHost<B,S>::update(s, mask);
}

template<class B, class S>
void bi::SparseStaticUpdater<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, const int p) {
  SparseStaticUpdaterHost<B,S>::update(s, mask, p);
}

#ifdef __CUDACC__
template<class B, class S>
void bi::SparseStaticUpdater<B,S>::update(State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask) {
  SparseStaticUpdaterGPU<B,S>::update(s, mask);
}

template<class B, class S>
void bi::SparseStaticUpdater<B,S>::update(State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask, const int p) {
  SparseStaticUpdaterGPU<B,S>::update(s, mask, p);
}
#endif

#endif
