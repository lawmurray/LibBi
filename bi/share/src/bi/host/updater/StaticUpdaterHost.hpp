/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICUPDATERHOST_HPP
#define BI_HOST_UPDATER_STATICUPDATERHOST_HPP

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
class StaticUpdaterHost {
public:
  static void update(State<B,ON_HOST>& s);

  static void update(State<B,ON_HOST>& s, const int p);
};
}

#include "StaticUpdaterVisitorHost.hpp"
#include "StaticUpdaterMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
void bi::StaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticUpdaterMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticUpdaterVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<
      block_is_matrix<S>::value,MatrixVisitor,ElementVisitor>::type Visitor;

  #pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(s, p, pax, x);
    }
  }
}

template<class B, class S>
void bi::StaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s, const int p) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticUpdaterMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef StaticUpdaterVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<
      block_is_matrix<S>::value,MatrixVisitor,ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(s, p, pax, x);
}

#endif
