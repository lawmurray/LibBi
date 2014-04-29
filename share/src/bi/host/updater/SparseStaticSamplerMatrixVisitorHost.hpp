/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICSAMPLERMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICSAMPLERMATRIXVISITORHOST_HPP

namespace bi {
/**
 * Visitor for SparseStaticSamplerHost.
 */
template<class B, class S, class PX, class OX>
class SparseStaticSamplerMatrixVisitorHost {
public:
  static void accept(Random& rng, State<B,ON_HOST>& s,
      const Mask<ON_HOST>& mask, const int p, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of SparseStaticSamplerMatrixVisitorHost.
 */
template<class B, class PX, class OX>
class SparseStaticSamplerMatrixVisitorHost<B,empty_typelist,PX,OX> {
public:
  static void accept(Random& rng, State<B,ON_HOST>& s,
      const Mask<ON_HOST>& mask, const int p, const PX& pax, OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
void bi::SparseStaticSamplerMatrixVisitorHost<B,S,PX,OX>::accept(Random& rng,
    State<B,ON_HOST>& s, const Mask<ON_HOST>& mask, const int p,
    const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;

  const int id = var_id<target_type>::value;

  if (mask.isDense(id)) {
    front::samples(rng, s, p, pax, x);
  } else if (mask.isSparse(id)) {
    BI_ASSERT_MSG(false, "Cannot do sparse update with matrix expression");
  }
  SparseStaticSamplerMatrixVisitorHost<B,pop_front,PX,OX>::accept(rng, s,
      mask, p, pax, x);
}

#endif
