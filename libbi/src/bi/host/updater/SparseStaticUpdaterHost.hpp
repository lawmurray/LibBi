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
#include "../../buffer/Mask.hpp"

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
#include "../../state/Pa.hpp"
#include "../../state/SpOx.hpp"
#include "../bind.hpp"

template<class B, class S>
void bi::SparseStaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef SpOx<ON_HOST,B,real,host> OX;
  typedef SparseStaticUpdaterVisitorHost<B,S,ON_HOST,PX,OX> Visitor;

  int p;
  PX pax;
  OX x;
  bind(s);

  #pragma omp parallel for private(p, pax)
  for (p = 0; p < s.size(); ++p) {
    Visitor::accept(mask, p, pax, x);
  }
  unbind(s);
}

template<class B, class S>
void bi::SparseStaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, const int p) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef SpOx<ON_HOST,B,real,host> OX;
  typedef SparseStaticUpdaterVisitorHost<B,S,ON_HOST,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(mask, p, pax, x);
  unbind(s);
}

#endif
