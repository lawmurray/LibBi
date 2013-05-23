/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICUPDATERHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICUPDATERHOST_HPP

#include "../../state/State.hpp"
#include "../../state/Mask.hpp"

namespace bi {
/**
 * Sparse static updater.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticUpdaterHost {
public:
  static void update(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask);

  static void update(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      const int p);
};
}

#include "SparseStaticUpdaterVisitorHost.hpp"
#include "SparseStaticUpdaterMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
void bi::SparseStaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef SparseStaticUpdaterMatrixVisitorHost<B,S,ON_HOST,PX,OX> MatrixVisitor;
  typedef SparseStaticUpdaterVisitorHost<B,S,ON_HOST,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

#pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

#pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(s, mask, p, pax, x);
    }
  }
}

template<class B, class S>
void bi::SparseStaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, const int p) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef SparseStaticUpdaterMatrixVisitorHost<B,S,ON_HOST,PX,OX> MatrixVisitor;
  typedef SparseStaticUpdaterVisitorHost<B,S,ON_HOST,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(s, mask, p, pax, x);
}

#endif
