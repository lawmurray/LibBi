/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SUPDATER_HPP
#define BI_UPDATER_SUPDATER_HPP

#include "../state/Static.hpp"
#include "../cuda/cuda.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for s-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH = STATIC_SHARED>
class SUpdater {
public:
  /**
   * Update s-net.
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

#include "SUpdateVisitor.hpp"
#include "../state/Pa.hpp"
#include "../host/bind.hpp"
#include "../host/host.hpp"
#include "../host/const_host.hpp"

template<class B, bi::StaticHandling SH>
void bi::SUpdater<B,SH>::update(Static<ON_HOST>& theta) {
  typedef typename B::STypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<B,real,pa,host,host,pa,host,host,host> V1;
  typedef real V2;
  typedef SUpdateVisitor<ON_HOST,B,S,V1> Visitor;

  if (net_size<B,S>::value > 0) {
    const int P = (SH == STATIC_SHARED) ? 1 : theta.size();
    bind(theta);
    #pragma omp parallel
    {
      int p;
      V1 pax(0);

      #pragma omp for
      for (p = 0; p < P; ++p) {
        pax.p = p;
        Visitor::accept(pax);
      }
    }
    unbind(theta);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/SUpdater.cuh"
#endif

#endif
