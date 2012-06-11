/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1622 $
 * $Date: 2011-06-13 22:28:52 +0800 (Mon, 13 Jun 2011) $
 */
#ifndef BI_HOST_UPDATER_STATICSAMPLERHOST_HPP
#define BI_HOST_UPDATER_STATICSAMPLERHOST_HPP

#include "../../random/Random.hpp"
#include "../../state/State.hpp"

namespace bi {
/**
 * Static sampler, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticSamplerHost {
public:
  /**
   * @copydoc StaticSampler::samples(Random&, State<B,ON_HOST>&)
   */
  static void samples(Random& rng, State<B,ON_HOST>& s);

  /**
   * @copydoc StaticSampler::samples(Random&, State<B,ON_HOST>&, const int)
   */
  static void samples(Random& rng, State<B,ON_HOST>& s, const int p);
};
}

#include "StaticSamplerVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
void bi::StaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s) {
  typedef Rng<ON_HOST> R1;
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticSamplerVisitorHost<B,S,R1,PX,OX> Visitor;

  bind(s);

  #pragma omp parallel
  {
    PX pax;
    OX x;
    R1& rng1 = rng.getHostRng();
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(rng1, p, pax, x);
    }
  }
  unbind(s);
}

template<class B, class S>
void bi::StaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const int p) {
  typedef Rng<ON_HOST> R1;
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef StaticSamplerVisitorHost<B,S,R1,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(rng.getHostRng(), p, pax, x);
  unbind(s);
}

#endif
