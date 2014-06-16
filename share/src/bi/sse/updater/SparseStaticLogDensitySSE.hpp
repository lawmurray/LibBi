/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 3021 $
 * $Date: 2012-08-31 17:27:15 +0800 (Fri, 31 Aug 2012) $
 */
#ifndef BI_SSE_UPDATER_SPARSESTATICLOGDENSITYSSE_HPP
#define BI_SSE_UPDATER_SPARSESTATICLOGDENSITYSSE_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Sparse static log-density evaluator, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticLogDensitySSE {
public:
  /**
   * @copydoc SparseStaticLogDensity::logDensities(State<B,ON_HOST>&, const Mask<ON_HOST>&, V1)
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      V1 lp);
};
}

#include "../sse_host.hpp"
#include "../../host/updater/SparseStaticLogDensityVisitorHost.hpp"
#include "../../host/updater/SparseStaticLogDensityMatrixVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensitySSE<B,S>::logDensities(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,sse_host,sse_host> PX;
  typedef Ou<ON_HOST,B,sse_host> OX;
  typedef SparseStaticLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  #pragma omp parallel
  {
    int p;
    PX pax;
    OX x;
    simd_real* lp1;

    #pragma omp for
    for (p = 0; p < s.size(); p += BI_SIMD_SIZE) {
      lp1 = reinterpret_cast<simd_real*>(&lp(p));
      Visitor::accept(mask, s, p, pax, x, *lp1);
    }
  }
}

#endif
