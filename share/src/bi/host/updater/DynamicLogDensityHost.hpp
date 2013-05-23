/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICLOGDENSITYHOST_HPP
#define BI_HOST_UPDATER_DYNAMICLOGDENSITYHOST_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Dynamic log-density evaluator, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicLogDensityHost {
public:
  /**
   * @copydoc DynamicLogDensity::logDensities(const T1, const T1, State<B,ON_HOST>&, V1)
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      V1 lp);

  /**
   * @copydoc DynamicLogDensity::logDensities(const T1, const T1, State<B,ON_HOST>&, const int, V1)
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, V1 lp);
};
}

#include "DynamicLogDensityVisitorHost.hpp"
#include "DynamicLogDensityMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensityHost<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef DynamicLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  #pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(t1, t2, s, p, pax, x, lp(p));
    }
  }
}

template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensityHost<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, const int p, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef DynamicLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(t1, t2, s, p, pax, x, lp(p));
}

#endif
