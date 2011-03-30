/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1244 $
 * $Date: 2011-01-31 10:37:29 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_UPDATER_ORUPDATER_HPP
#define BI_UPDATER_ORUPDATER_HPP

#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "../misc/Pipelineable.hpp"
#include "../math/locatable.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for or-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 */
template<class B>
class ORUpdater :
    public Pipelineable<typename host_matrix_temp_type<real>::type> {
public:
  /**
   * Constructor.
   *
   * @param rng Random number generator.
   */
  ORUpdater(Random& rng);

  /**
   * Update r-net.
   *
   * @param s State to update.
   */
  void update(State<ON_HOST>& s);

  /**
   * @copydoc update(Static<ON_HOST>&)
   */
  void update(State<ON_DEVICE>& s);

private:
  /**
   * Random number generator.
   */
  Random& rng;
};
}

#include "ORUpdateVisitor.hpp"

template<class B>
bi::ORUpdater<B>::ORUpdater(Random& rng) : rng(rng) {
  //
}

template<class B>
void bi::ORUpdater<B>::update(State<ON_HOST>& s) {
  typedef typename B::OTypeList S;
  typedef BOOST_TYPEOF(s.get(OR_NODE)) M1;
  typedef ORUpdateVisitor<B,S,M1> Visitor;

  Visitor::accept(rng, s.get(OR_NODE));
}

#ifdef __CUDACC__
#include "../cuda/updater/ORUpdater.cuh"
#endif

#endif
