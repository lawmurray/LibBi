/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICMAXLOGDENSITYHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICMAXLOGDENSITYHOST_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Sparse static maximum log-density evaluator, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticMaxLogDensityHost {
public:
  /**
   * @copydoc SparseStaticMaxLogDensity::maxLogDensities(State<B,ON_HOST>&, const Mask<ON_HOST>&, V1)
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask, V1 lp);

  /**
   * @copydoc SparseStaticMaxLogDensity::maxLogDensities(State<B,ON_HOST>&, const int, const Mask<ON_HOST>&, V1)
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, const int p, const Mask<ON_HOST>& mask, V1 lp);
};
}

#include "SparseStaticMaxLogDensityVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask, V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef SparseStaticMaxLogDensityVisitorHost<B,S,PX,OX> Visitor;

  bind(s);

  #pragma omp parallel
  {
    PX pax;
    OX x;
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(mask, p, pax, x, lp(p));
    }
  }
  unbind(s);
}

template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensityHost<B,S>::maxLogDensities(State<B,ON_HOST>& s, const int p, const Mask<ON_HOST>& mask, V1 lp) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef SparseStaticMaxLogDensityVisitorHost<B,S,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(mask, p, pax, x, lp(p));
  unbind(s);
}

#endif
