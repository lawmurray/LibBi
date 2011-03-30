/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_PUPDATER_CUH
#define BI_CUDA_UPDATER_PUPDATER_CUH

#include "../../host/bind.hpp"

template<class B>
void bi::PUpdater<B>::update(Static<ON_DEVICE>& theta) {
  typedef typename B::PTypeList S;

  if (net_size<B,S>::value > 0) {
    bind(theta); // copies to constant memory etc
  }
}

#endif
