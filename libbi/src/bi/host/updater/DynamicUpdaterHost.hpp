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
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterHost<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-conditions */
  assert (t1 <= t2);

  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> Visitor;

  int p;
  PX pax;
  OX x;
  bind(s);

  #pragma omp parallel for private(p)
  for (p = 0; p < s.size(); ++p) {
    Visitor::accept(t1, t2, p, pax, x);
  }
  unbind(s);
}

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterHost<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, const int p) {
  /* pre-conditions */
  assert (t1 <= t2);

  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef DynamicUpdaterVisitorHost<B,S,T1,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(t1, t2, p, pax, x);
  unbind(s);
}

#endif
