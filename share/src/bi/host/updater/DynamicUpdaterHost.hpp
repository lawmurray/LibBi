/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICUPDATERHOST_HPP
#define BI_HOST_UPDATER_DYNAMICUPDATERHOST_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Dynamic updater.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicUpdaterHost {
public:
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);

  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p);
};
}

#include "DynamicUpdaterVisitorHost.hpp"
#include "DynamicUpdaterMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

#include "boost/mpl/if.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterHost<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicUpdaterMatrixVisitorHost<B,S,T1,PX,OX> MatrixVisitor;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

#pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(t1, t2, s, p, pax, x);
    }
  }
}

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterHost<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, const int p) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicUpdaterMatrixVisitorHost<B,S,T1,PX,OX> MatrixVisitor;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(t1, t2, s, p, pax, x);
}

#endif
