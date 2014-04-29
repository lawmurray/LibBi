/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICUPDATERVISITORHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICUPDATERVISITORHOST_HPP

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
class SparseStaticUpdaterVisitorHost {
public:
  /**
   * Update.
   *
   * @param mask Mask.
   * @param p Trajectory id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static void accept(State<B,ON_HOST>& s, const Mask<L>& mask, const int p,
      const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of SparseStaticUpdaterVisitorHost.
 */
template<class B, Location L, class PX, class OX>
class SparseStaticUpdaterVisitorHost<B,empty_typelist,L,PX,OX> {
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
inline void bi::SparseStaticUpdaterVisitorHost<B,S,L,PX,OX>::accept(
    State<B,ON_HOST>& s, const Mask<L>& mask, const int p, const PX& pax,
    OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename front::coord_type coord_type;

  const int id = var_id<target_type>::value;
  int ix = 0;
  coord_type cox;

  if (mask.isDense(id)) {
    while (ix < mask.getSize(id)) {
      front::simulates(s, p, ix, cox, pax, x);
      ++cox;
      ++ix;
    }
  } else if (mask.isSparse(id)) {
    while (ix < mask.getSize(id)) {
      cox.setIndex(mask.getIndex(id, ix));
      front::simulates(s, p, ix, cox, pax, x);
      ++ix;
    }
  }

  SparseStaticUpdaterVisitorHost<B,pop_front,L,PX,OX>::accept(s, mask, p, pax,
      x);
}

#endif
