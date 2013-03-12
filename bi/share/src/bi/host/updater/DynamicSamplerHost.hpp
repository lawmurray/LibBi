/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICSAMPLERHOST_HPP
#define BI_HOST_UPDATER_DYNAMICSAMPLERHOST_HPP

#include "../../random/Random.hpp"
#include "../../state/State.hpp"

namespace bi {
/**
 * Dynamic sampler, on host.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicSamplerHost {
public:
  /**
   * @copydoc DynamicSampler::samples(Random&, const T1, const T1, State<B,ON_HOST>&)
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_HOST>& s);

  /**
   * @copydoc DynamicSampler::samples(Random&, const T1, const T1, State<B,ON_HOST>&, const int)
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_HOST>& s, const int p);
};
}

#include "DynamicSamplerVisitorHost.hpp"
#include "DynamicSamplerMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicSamplerHost<B,S>::samples(Random& rng, const T1 t1,
    const T1 t2, State<B,ON_HOST>& s) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicSamplerMatrixVisitorHost<B,S,R1,PX,OX> MatrixVisitor;
  typedef DynamicSamplerVisitorHost<B,S,R1,PX,OX> ElementVisitor;
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
      Visitor::accept(rng1, t1, t2, s, p, pax, x);
    }
  }
}

template<class B, class S>
template<class T1>
void bi::DynamicSamplerHost<B,S>::samples(Random& rng, const T1 t1,
    const T1 t2, State<B,ON_HOST>& s, const int p) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef DynamicSamplerMatrixVisitorHost<B,S,R1,PX,OX> MatrixVisitor;
  typedef DynamicSamplerVisitorHost<B,S,R1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(rng.getHostRng(), t1, t2, s, p, pax, x);
}

#endif
