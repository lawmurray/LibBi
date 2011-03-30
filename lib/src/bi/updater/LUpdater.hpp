/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_LUPDATER_HPP
#define BI_UPDATER_LUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * Calculator for likelihood of states given observations.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 */
template<class B>
class LUpdater {
public:
  /**
   * Update log-likelihood weights.
   *
   * @tparam V1 Integer vector type.
   * @tparam V2 Floating point vector type.
   *
   * @param s State.
   * @param ids Ids of o-nodes to consider.
   * @param[in,out] lws Log-weights.
   */
  template<class V1, class V2>
  void update(State<ON_HOST>& s, const V1& ids, V2& lws);

  /**
   * @copydoc update(State<ON_HOST>&, const V1&, V2&)
   */
  template<class V1, class V2>
  void update(State<ON_DEVICE>& s, const V1& ids, V2& lws);
};
}

#include "LUpdateVisitor.hpp"
#include "../state/Pa.hpp"
#include "../host/host.hpp"
#include "../host/const_host.hpp"

template<class B>
template<class V1, class V2>
void bi::LUpdater<B>::update(State<ON_HOST>& s, const V1& ids, V2& lws) {
  typedef typename B::OTypeList S;
  typedef Pa<B,real,const_host,host,host,const_host,host,host,host> V3;
  typedef LUpdateVisitor<ON_HOST,B,S,V1,V3,real> Visitor;

  const int P = lws.size();

  if (ids.size() > 0) {
    bind(s);
    #pragma omp parallel
    {
      real l;
      int p;
      V3 pax(0);

      #pragma omp for
      for (p = 0; p < P; ++p) {
        pax.p = p;
        l = 0.0;
        Visitor::accept(ids, 0, pax, l);
        lws(p) += l;
      }
    }
    unbind(s);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/LUpdater.cuh"
#endif

#endif
