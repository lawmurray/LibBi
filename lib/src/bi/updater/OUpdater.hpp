/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_OUPDATER_HPP
#define BI_UPDATER_OUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for o-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH = STATIC_SHARED>
class OUpdater {
public:
  /**
   * Update o-net.
   *
   * @tparam Integral vector type.
   *
   * @param ids Ids of o-nodes to be updated.
   * @param s State to update.
   */
  template<class V1>
  void update(const V1& ids, State<ON_HOST>& s);

  /**
   * @copydoc update(const V1&, State<ON_HOST>)
   */
  template<class V1>
  void update(const V1& ids, State<ON_DEVICE>& s);
};
}

#include "OUpdateVisitor.hpp"
#include "../state/Pa.hpp"
#include "../host/host.hpp"
#include "../host/const_host.hpp"

template<class B, bi::StaticHandling SH>
template<class V1>
void bi::OUpdater<B,SH>::update(const V1& ids, State<ON_HOST>& s) {
  typedef typename B::OTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<B,real,pa,host,host,pa,host,host,host> V2;
  typedef real V3;
  typedef OUpdateVisitor<ON_HOST,B,S,V1,V2,V3> Visitor;

  if (ids.size() > 0) {
    const int P = s.size();
    bind(s);
    #pragma omp parallel
    {
      int i, p;
      V2 pax(0);
      V3 xnxt[ids.size()];

      #pragma omp for
      for (p = 0; p < P; ++p) {
        pax.p = p;
        Visitor::accept(ids, pax, xnxt);
        for (i = 0; i < ids.size(); ++i) {
          s.get(O_NODE)(pax.p, i) = xnxt[i];
        }
      }
    }
    unbind(s);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/OUpdater.cuh"
#endif

#endif
