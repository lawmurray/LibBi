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
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
void bi::StaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticUpdaterVisitorHost<B,S,PX,OX> Visitor;

  int p;
  PX pax;
  OX x;
  bind(s);

  #pragma omp parallel for private(p, pax)
  for (p = 0; p < s.size(); ++p) {
    Visitor::accept(p, pax, x);
  }
  unbind(s);
}

template<class B, class S>
void bi::StaticUpdaterHost<B,S>::update(State<B,ON_HOST>& s, const int p) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticUpdaterVisitorHost<B,S,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(p, pax, x);
  unbind(s);
}

#endif
