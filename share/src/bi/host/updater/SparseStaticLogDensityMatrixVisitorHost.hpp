/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICLOGDENSITYMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICLOGDENSITYMATRIXVISITORHOST_HPP

namespace bi {
/**
 * Visitor for SparseStaticLogDensityHost.
 */
template<class B, class S, class PX, class OX>
class SparseStaticLogDensityMatrixVisitorHost {
public:
  template<class T1>
  static void accept(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      const int p, const PX& pax, OX& x, T1& lp);
};

/**
 * @internal
 *
 * Base case of SparseStaticLogDensityMatrixVisitorHost.
 */
template<class B, class PX, class OX>
class SparseStaticLogDensityMatrixVisitorHost<B,empty_typelist,PX,OX> {
public:
  template<class T1>
  static void accept(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      const int p, const PX& pax, OX& x, T1& lp) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
template<class T1>
void bi::SparseStaticLogDensityMatrixVisitorHost<B,S,PX,OX>::accept(
    State<B,ON_HOST>& s, const Mask<ON_HOST>& mask, const int p,
    const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;

  const int id = var_id<target_type>::value;

  if (mask.isDense(id)) {
    front::logDensities(s, p, pax, x, lp);
  } else if (mask.isSparse(id)) {
    BI_ASSERT_MSG(false, "Cannot do sparse update with matrix expression");
  }

  SparseStaticLogDensityMatrixVisitorHost<B,pop_front,PX,OX>::accept(s, mask,
      p, pax, x, lp);
}

#endif
