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
   * @param t Current time.
   * @param tnxt Time to which to step forward.
   * @param[out] s State.
   */
  static void accept(Random& rng, const real t, const real tnxt, M1& s);
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
  static void accept(Random& rng, const real t, const real tnxt, M1& s) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/random_traits.hpp"
#include "../misc/assert.hpp"

template<class B, class S, class M1>
inline void bi::RUpdateVisitor<B,S,M1>::accept(Random& rng, const real t,
    const real tnxt, M1& s) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  static const int start = node_start<B,front>::value;
  static const int size = node_size<B,front>::value;

  if (is_uniform_variate<front>::value) {
    rng.uniforms(vec(columns(s, start, size)), -0.5, 0.5);
  } else if (is_gaussian_variate<front>::value) {
    rng.gaussians(vec(columns(s, start, size)));
  } else if (is_wiener_increment<front>::value) {
    if (std::abs(tnxt - t) > 0.0) {
      rng.gaussians(vec(columns(s, start, size)), 0.0, std::sqrt(std::abs(tnxt - t)));
    } else {
      columns(s, start, size).clear();
    }
  } else {
    BI_ASSERT(false, "Random variate has unsupported distribution");
  }

  RUpdateVisitor<B,pop_front,M1>::accept(rng, t, tnxt, s);
}

#endif
