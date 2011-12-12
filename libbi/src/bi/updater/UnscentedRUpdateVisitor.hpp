/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_UNSCENTEDRUPDATEVISITOR_HPP
#define BI_UPDATER_UNSCENTEDRUPDATEVISITOR_HPP

#include "../typelist/typelist.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * Specialised visitor for updating r-net, accompanies UnscentedRUpdater.
 *
 * @tparam B Model type.
 * @tparam S Type list for r-net.
 * @tparam V1 Vector type.
 */
template<class B, class S, class V1>
class UnscentedRUpdateVisitor {
public:
  /**
   * Update r-net.
   *
   * @param t Current time.
   * @param tnxt Time to which to step forward.
   * @param a Scaling factor for variates.
   * @param[out] x1 Diagonal of state to write positive variates.
   * @param[out] x2 Diagonal of state to write negative variates.
   */
  static void accept(const real t, const real tnxt, const real a, V1& x1,
      V1& x2);
};

/**
 * @internal
 *
 * Specialised base case of UnscentedRUpdateVisitor.
 *
 * @tparam B Model type.
 * @tparam M1 Matrix type.
 */
template<class B, class V1>
class UnscentedRUpdateVisitor<B,empty_typelist,V1> {
public:
  static void accept(const real t, const real tnxt, const real a, V1& x1,
      V1& x2) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/random_traits.hpp"
#include "../misc/assert.hpp"

template<class B, class S, class V1>
inline void bi::UnscentedRUpdateVisitor<B,S,V1>::accept(const real t,
    const real tnxt, const real a, V1& x1, V1& x2) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef UnscentedRUpdateVisitor<B,pop_front,V1> Visitor;

  static const int start = node_start<B,front>::value;
  static const int size = node_size<B,front>::value;

  BOOST_AUTO(d1, subrange(x1, start, size));
  BOOST_AUTO(d2, subrange(x2, start, size));
  if (is_gaussian_variate<front>::value) {
    bi::fill(d1.begin(), d1.end(), a);
    bi::fill(d2.begin(), d2.end(), -a);
  } else if (is_uniform_variate<front>::value) {
    bi::fill(d1.begin(), d1.end(), 0.5 + a*std::sqrt(1.0/12.0));
    bi::fill(d2.begin(), d2.end(), 0.5 - a*std::sqrt(1.0/12.0));
  } else if (is_wiener_increment<front>::value) {
    if (tnxt - t > 0.0) {
      bi::fill(d1.begin(), d1.end(), a*std::sqrt(tnxt - t));
      bi::fill(d2.begin(), d2.end(), -a*std::sqrt(tnxt - t));
    } else {
      d1.clear();
      d2.clear();
    }
  } else {
    BI_ASSERT(false, "Random variate has unsupported distribution");
  }

  Visitor::accept(t, tnxt, a, x1, x2);
}

#endif
