/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_RUPDATEVISITOR_HPP
#define BI_UPDATER_RUPDATEVISITOR_HPP

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
class RUpdateVisitor {
public:
  /**
   * Update r-net.
   *
   * @param rng Random number generator.
   * @param[out] s State.
   */
  static void accept(Random& rng, M1& s);
};

/**
 * @internal
 *
 * Specialised base case of RUpdateVisitor.
 *
 * @tparam B Model type.
 * @tparam M1 Matrix type.
 */
template<class B, class M1>
class RUpdateVisitor<B,empty_typelist,M1> {
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
inline void bi::RUpdateVisitor<B,S,M1>::accept(Random& rng, M1& s) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  Coord cox;
  int id = node_start<B,front>::value;
  for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
    for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
      for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
        if (is_uniform_variate<front>::value) {
          rng.uniforms(column(s, id));
        } else if (is_gaussian_variate<front>::value) {
          rng.gaussians(column(s, id));
        } else {
          BI_ASSERT(false, "Random variate has unsupported distribution");
        }
      }
    }
  }
  RUpdateVisitor<B,pop_front,M1>::accept(rng, s);
}

#endif
