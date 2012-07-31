/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_SPARSESTATICMAXLOGDENSITYVISITORHOST_HPP
#define BI_HOST_UPDATER_SPARSESTATICMAXLOGDENSITYVISITORHOST_HPP

namespace bi {
/**
 * Visitor for SparseStaticMaxLogDensityHost.
 */
template<class B, class S, class PX, class OX>
class SparseStaticMaxLogDensityVisitorHost {
public:
  template<class T1>
  static void accept(const Mask<ON_HOST>& mask, const int p, const PX& pax, OX& x, T1& lp);
};

/**
 * @internal
 *
 * Base case of SparseStaticMaxLogDensityVisitorHost.
 */
template<class B, class PX, class OX>
class SparseStaticMaxLogDensityVisitorHost<B,empty_typelist,PX,OX> {
public:
  template<class T1>
  static void accept(const Mask<ON_HOST>& mask, const int p, const PX& pax, OX& x, T1& lp) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/target_traits.hpp"

template<class B, class S, class PX, class OX>
template<class T1>
void bi::SparseStaticMaxLogDensityVisitorHost<B,S,PX,OX>::accept(const Mask<ON_HOST>& mask, const int p, const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename target_type::coord_type coord_type;

  const int id = var_id<target_type>::value;
  int ix = 0;
  coord_type cox;

  if (mask.isDense(id)) {
    while (ix < target_size<target_type>::value) {
      front::maxLogDensities(p, ix, cox, pax, x, lp);
      ++cox;
      ++ix;
    }
  } else if (mask.isSparse(id)) {
    while (ix < mask.getSize(id)) {
      cox.setIndex(mask.getIndex(id, ix));
      front::maxLogDensities(p, ix, cox, pax, x, lp);
      ++ix;
    }
  }

  SparseStaticMaxLogDensityVisitorHost<B,pop_front,PX,OX>::accept(mask, p, pax, x, lp);
}

#endif
