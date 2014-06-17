/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
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
#include "../../host/updater/DynamicUpdaterVisitorHost.hpp"
#include "../../host/updater/DynamicUpdaterMatrixVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterSSE<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  BI_ASSERT(t1 <= t2);

  typedef Pa<ON_HOST,B,host,host,sse_host,sse_host> PX;
  typedef Ou<ON_HOST,B,sse_host> OX;
  typedef DynamicUpdaterMatrixVisitorHost<B,S,T1,PX,OX> MatrixVisitor;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  #pragma omp parallel
  {
    int p;
    PX pax;
    OX x;

    #pragma omp for
    for (p = 0; p < s.size(); p += BI_SIMD_SIZE) {
      Visitor::accept(t1, t2, s, p, pax, x);
    }
  }
}

#endif
