/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICMAXLOGDENSITYHOST_HPP
#define BI_HOST_UPDATER_STATICMAXLOGDENSITYHOST_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Static maximum log-density evaluator, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticMaxLogDensityHost {
public:
  /**
   * @copydoc StaticMaxLogDensity::maxLogDensities(State<B,ON_HOST>&, V1)
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, V1 lp);

  /**
   * @copydoc StaticMaxLogDensity::maxLogDensities(State<B,ON_HOST>&, const int, V1)
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, const int p, V1 lp);
};
}

#include "StaticMaxLogDensityVisitorHost.hpp"
#include "StaticMaxLogDensityMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticMaxLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticMaxLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

#pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

#pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(s, p, pax, x, lp(p));
    }
  }
}

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    const int p, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticMaxLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticMaxLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(s, p, pax, x, lp(p));
}

#endif
