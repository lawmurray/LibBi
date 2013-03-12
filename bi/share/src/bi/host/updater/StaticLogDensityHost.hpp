/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICLOGDENSITYHOST_HPP
#define BI_HOST_UPDATER_STATICLOGDENSITYHOST_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Static log-density evaluator, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticLogDensityHost {
public:
  /**
   * @copydoc StaticLogDensity::logDensities(State<B,ON_HOST>&, V1)
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, V1 lp);

  /**
   * @copydoc StaticLogDensity::logDensities(State<B,ON_HOST>&, const int, V1)
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, const int p, V1 lp);
};
}

#include "StaticLogDensityVisitorHost.hpp"
#include "StaticLogDensityMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class V1>
void bi::StaticLogDensityHost<B,S>::logDensities(State<B,ON_HOST>& s, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
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
void bi::StaticLogDensityHost<B,S>::logDensities(State<B,ON_HOST>& s,
    const int p, V1 lp) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticLogDensityMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticLogDensityVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(s, p, pax, x, lp(p));
}

#endif
