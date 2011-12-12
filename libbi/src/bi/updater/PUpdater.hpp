/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_PUPDATER_HPP
#define BI_UPDATER_PUPDATER_HPP

#include "../state/Static.hpp"
#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Updater for p-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 */
template<class B>
class PUpdater {
public:
  /**
   * Update p-net.
   *
   * @param theta Static state to update.
   */
  void update(Static<ON_HOST>& theta);

  /**
   * @copydoc update(Static<ON_HOST>&)
   */
  void update(Static<ON_DEVICE>& theta);
};
}

template<class B>
void bi::PUpdater<B>::update(Static<ON_HOST>& theta) {
  //
}

#ifdef __CUDACC__
#include "../cuda/updater/PUpdater.cuh"
#endif

#endif
