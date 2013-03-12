/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
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
#include "SparseStaticSamplerMatrixVisitorHost.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"
#include "../../traits/block_traits.hpp"

template<class B, class S>
void bi::SparseStaticSamplerHost<B,S>::samples(Random& rng,
    State<B,ON_HOST>& s, const Mask<ON_HOST>& mask) {
  typedef RngHost R1;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef SparseStaticSamplerMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticSamplerVisitorHost<B,S,PX,OX> ElementVisitor;
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
      Visitor::accept(rng, s, mask, p, pax, x);
    }
  }
}

template<class B, class S>
void bi::SparseStaticSamplerHost<B,S>::samples(Random& rng,
    State<B,ON_HOST>& s, const int p, const Mask<ON_HOST>& mask) {
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef Ou<ON_HOST,B,host> OX;
  typedef SparseStaticSamplerMatrixVisitorHost<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticSamplerVisitorHost<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  PX pax;
  OX x;
  Visitor::accept(rng.getHostRng(), s, mask, p, pax, x);
}

#endif
