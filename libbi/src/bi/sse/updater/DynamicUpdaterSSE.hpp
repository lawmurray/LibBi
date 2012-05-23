/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2138 $
 * $Date: 2011-11-11 14:55:42 +0800 (Fri, 11 Nov 2011) $
 */
#ifndef BI_SSE_UPDATER_DYNAMICUPDATERSSE_HPP
#define BI_SSE_UPDATER_DYNAMICUPDATERSSE_HPP

namespace bi {
/**
 * Dynamic updater, using SSE instructions.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicUpdaterSSE {
public:
  /**
   * @copydoc DynamicUpdater::update()
   */
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);
};

}

#include "../sse_host.hpp"
#include "../sse_const_host.hpp"
#include "../../state/Pa.hpp"
#include "../../host/updater/DynamicUpdaterVisitorHost.hpp"
#include "../../host/bind.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterSSE<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  assert (t1 <= t2);

  typedef Pa<ON_HOST,B,sse_real,sse_const_host,sse_host,sse_host,sse_host> PX;
  typedef Ox<ON_HOST,B,sse_real,sse_host> OX;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> Visitor;

  int p;
  PX pax;
  OX x;
  bind(s);

  #pragma omp parallel for private(p, pax)
  for (p = 0; p < s.size(); p += BI_SSE_SIZE) {
    Visitor::accept(t1, t2, p, pax, x);
  }
  unbind(s);
}

#endif
