/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1622 $
 * $Date: 2011-06-13 22:28:52 +0800 (Mon, 13 Jun 2011) $
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICSAMPLERHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICSAMPLERHOST_HPP

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
class SparseStaticSamplerHost {
public:
  /**
   * @copydoc SparseStaticSampler::samples(Random&, State<B,ON_HOST>&, const Mask<ON_HOST>&)
   */
  static void samples(Random& rng, State<B,ON_HOST>& s,
      const Mask<ON_HOST>& mask);

  /**
   * @copydoc SparseStaticSampler::samples(Random&, State<B,ON_HOST>&, const int, const Mask<ON_HOST>&)
   */
  static void samples(Random& rng, State<B,ON_HOST>& s, const int p,
      const Mask<ON_HOST>& mask);
};
}

#include "SparseStaticSamplerVisitorHost.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"
#include "../bind.hpp"

template<class B, class S>
void bi::SparseStaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef SparseStaticSamplerVisitorHost<B,S,PX,OX> Visitor;

  bind(s);

  #pragma omp parallel
  {
    PX pax;
    OX x;
    R1& rng1 = rng.getHostRng();
    int p;

    #pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(rng, mask, p, pax, x);
    }
  }
  unbind(s);
}

template<class B, class S>
void bi::SparseStaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const int p, const Mask<ON_HOST>& mask) {
  typedef Pa<ON_HOST,B,real,const_host,host,host,host> PX;
  typedef Ox<ON_HOST,B,real,host> OX;
  typedef SparseStaticSamplerVisitorHost<B,S,PX,OX> Visitor;

  PX pax;
  OX x;
  bind(s);
  Visitor::accept(rng.getHostRng(), mask, p, pax, x);
  unbind(s);
}

#endif
