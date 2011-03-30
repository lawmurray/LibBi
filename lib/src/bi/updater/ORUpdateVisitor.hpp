/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1235 $
 * $Date: 2011-01-28 17:51:09 +0800 (Fri, 28 Jan 2011) $
 */
#ifndef BI_UPDATER_ORUPDATEVISITOR_HPP
#define BI_UPDATER_ORUPDATEVISITOR_HPP

#include "../typelist/typelist.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for updating r-net on host.
 *
 * @tparam B Model type.
 * @tparam S Type list for r-net.
 * @tparam M1 Matrix type.
 */
template<class B, class S, class M1>
class ORUpdateVisitor {
public:
  /**
   * Update or-net.
   *
   * @param rng Random number generator.
   * @param[out] s State.
   */
  static void accept(Random& rng, M1& s);
};

/**
 * @internal
 *
 * Specialised base case of ORUpdateVisitor.
 *
 * @tparam B Model type.
 * @tparam M1 Matrix type.
 */
template<class B, class M1>
class ORUpdateVisitor<B,empty_typelist,M1> {
public:
  static void accept(Random& rng, M1& s) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/random_traits.hpp"
#include "../misc/assert.hpp"

template<class B, class S, class M1>
inline void bi::ORUpdateVisitor<B,S,M1>::accept(Random& rng, M1& s) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  Coord cox;
  int id = node_start<B,front>::value;
  for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
    for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
      for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
        if (is_gaussian_likelihood<front>::value ||
            is_log_normal_likelihood<front>::value) {
          rng.gaussians(column(s, id).buf());
        } else {
          BI_ASSERT(false, "Random variate has unsupported distribution");
        }
      }
    }
  }
  ORUpdateVisitor<B,pop_front,M1>::accept(rng, s);
}

#endif
