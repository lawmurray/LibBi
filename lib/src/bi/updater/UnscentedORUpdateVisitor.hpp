/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_UNSCENTEDORUPDATEVISITOR_HPP
#define BI_UPDATER_UNSCENTEDORUPDATEVISITOR_HPP

#include "../typelist/typelist.hpp"
#include "../state/State.hpp"

namespace bi {
/**
 * @internal
 *
 * Specialised visitor for updating or-net, accompanies UnscentedORUpdater.
 *
 * @tparam B Model type.
 * @tparam S Type list for r-net.
 * @tparam V1 Vector type.
 */
template<class B, class S, class V1>
class UnscentedORUpdateVisitor {
public:
  /**
   * Update or-net.
   *
   * @param id Variable id.
   * @param a Scaling factor for variates.
   * @param[out] x1 Diagonal of state to write positive variates.
   * @param[out] x2 Diagonal of state to write negative variates.
   */
  static void accept(const int id, const real a, V1 x1, V1 x2);
};

/**
 * @internal
 *
 * Specialised base case of UnscentedORUpdateVisitor.
 *
 * @tparam B Model type.
 * @tparam M1 Matrix type.
 */
template<class B, class V1>
class UnscentedORUpdateVisitor<B,empty_typelist,V1> {
public:
  static void accept(const int id, const real a, V1 x1, V1 x2) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/likelihood_traits.hpp"
#include "../misc/assert.hpp"

template<class B, class S, class V1>
inline void bi::UnscentedORUpdateVisitor<B,S,V1>::accept(const int id,
    const real a, V1 x1, V1 x2) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef UnscentedORUpdateVisitor<B,pop_front,V1> Visitor;

  if (id == node_id<B,front>::value) {
    if (is_gaussian_likelihood<front>::value || is_log_normal_likelihood<front>::value) {
      bi::fill(x1.begin(), x1.end(), a);
      bi::fill(x2.begin(), x2.end(), -a);
    } else {
      BI_ASSERT(false, "Likelihood has unsupported type");
    }
  } else {
    Visitor::accept(id, a, x1, x2);
  }
}

#endif
