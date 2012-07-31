/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1622 $
 * $Date: 2011-06-13 22:28:52 +0800 (Mon, 13 Jun 2011) $
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
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticMaxLogDensityVisitorHost<B,S,PX,OX> Visitor;

  bind(s);

  #pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(p, pax, x, lp(p));
    }
  }
  unbind(s);
}

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    const int p, V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticMaxLogDensityVisitorHost<B,S,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(p, pax, x, lp(p));
  unbind(s);
}

#endif
