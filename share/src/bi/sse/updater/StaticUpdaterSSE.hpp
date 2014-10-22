/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 3021 $
 * $Date: 2012-08-31 17:27:15 +0800 (Fri, 31 Aug 2012) $
 */
#ifndef BI_SSE_UPDATER_STATICUPDATERSSE_HPP
#define BI_SSE_UPDATER_STATICUPDATERSSE_HPP

#include "../../state/State.hpp"

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
class StaticUpdaterSSE {
public:
  static void update(State<B,ON_HOST>& s);
};
}

#include "../sse_host.hpp"
#include "../../host/updater/StaticUpdaterVisitorHost.hpp"
#include "../../host/updater/StaticUpdaterMatrixVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
void bi::StaticUpdaterSSE<B,S>::update(State<B,ON_HOST>& s) {
  typedef Pa<ON_HOST,B,host,host,sse_host,sse_host> PX;
  typedef Ou<ON_HOST,B,sse_host> OX;
  typedef StaticUpdaterMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticUpdaterVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

#pragma omp parallel
  {
    int p;
    PX pax;
    OX x;

#pragma omp for
    for (p = 0; p < s.size(); p += BI_SIMD_SIZE) {
      Visitor::accept(s, p, pax, x);
    }
  }
}

#endif
