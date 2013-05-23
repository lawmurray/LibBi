/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
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
#include "StaticSamplerMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
void bi::StaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticSamplerMatrixVisitorHost<B,S,R1,PX,OX> MatrixVisitor;
  typedef StaticSamplerVisitorHost<B,S,R1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

#pragma omp parallel
  {
    PX pax;
    OX x;
    R1& rng1 = rng.getHostRng();
    int p;

#pragma omp for
    for (p = 0; p < s.size(); ++p) {
      Visitor::accept(rng1, s, p, pax, x);
    }
  }
}

template<class B, class S>
void bi::StaticSamplerHost<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const int p) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef StaticSamplerMatrixVisitorHost<B,S,R1,PX,OX> MatrixVisitor;
  typedef StaticSamplerVisitorHost<B,S,R1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(rng.getHostRng(), s, p, pax, x);
}

#endif
