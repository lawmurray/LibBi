/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICUPDATERMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICUPDATERMATRIXVISITORHOST_HPP

#include "../../typelist/typelist.hpp"
#include "../../state/Mask.hpp"

namespace bi {
/**
 * Visitor for sparse static updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam L Location.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, Location L, class PX, class OX>
class SparseStaticUpdaterMatrixVisitorHost {
public:
  static void accept(State<B,ON_HOST>& s, const Mask<L>& mask, const int p,
      const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of SparseStaticUpdaterMatrixVisitorHost.
 */
template<class B, Location L, class PX, class OX>
class SparseStaticUpdaterMatrixVisitorHost<B,empty_typelist,L,PX,OX> {
public:
  static void accept(State<B,ON_HOST>& s, const Mask<L>& mask, const int p,
      const PX& pax, OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/var_traits.hpp"

template<class B, class S, bi::Location L, class PX, class OX>
inline void bi::SparseStaticUpdaterMatrixVisitorHost<B,S,L,PX,OX>::accept(
    State<B,ON_HOST>& s, const Mask<L>& mask, const int p, const PX& pax,
    OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;

  const int id = var_id<target_type>::value;

  if (mask.isDense(id)) {
    front::simulates(s, p, pax, x);
  } else if (mask.isSparse(id)) {
    BI_ASSERT_MSG(false, "Cannot do sparse update with matrix expression");
  }
  SparseStaticUpdaterMatrixVisitorHost<B,pop_front,L,PX,OX>::accept(s, mask,
      p, pax, x);
}

#endif
