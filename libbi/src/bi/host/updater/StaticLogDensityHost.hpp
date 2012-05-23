/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1622 $
 * $Date: 2011-06-13 22:28:52 +0800 (Mon, 13 Jun 2011) $
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
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
template<class V1>
void bi::StaticLogDensityHost<B,S>::logDensities(State<B,ON_HOST>& s, V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticLogDensityVisitorHost<B,S,PX,OX> Visitor;

  int p;
  PX pax;
  OX x;
  bind(s);

  #pragma omp parallel for private(p)
  for (p = 0; p < s.size(); ++p) {
    Visitor::accept(p, pax, x, lp(p));
  }
  unbind(s);
}

template<class B, class S>
template<class V1>
void bi::StaticLogDensityHost<B,S>::logDensities(State<B,ON_HOST>& s, const int p,
    V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticLogDensityVisitorHost<B,S,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(p, pax, x, lp(p));
  unbind(s);
}

#endif
