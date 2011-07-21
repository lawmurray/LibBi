/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_RUPDATER_HPP
#define BI_UPDATER_RUPDATER_HPP

#include "../state/State.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for r-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam CL Cache location.
 */
template<class B, Location CL = ON_HOST>
class RUpdater {
public:
  /**
   * Matrix type.
   */
  typedef typename locatable_temp_matrix<CL,real>::type matrix_type;

  /**
   * Constructor.
   *
   * @param rng Random number generator.
   */
  RUpdater(Random& rng);

  /**
   * Update r-net.
   *
   * @param t Current time.
   * @param tnxt Time to which to step forward.
   * @param s State to update.
   */
  template<Location L>
  void update(const real t, const real tnxt, State<L>& s);

  /**
   * Skip next update(s).
   *
   * @param nupdates Number of updates to skip.
   *
   * This is useful for methods which adjust random variates externally.
   */
  void skipNext(const int nupdates = 1);

  /**
   * Use buffer for next update(s).
   *
   * @param nupdates Number of updates to set.
   *
   * Use buf() to retrieve and fill the buffer with an appropriate number of
   * updates before this call. A check is made to ensure that the buffer is
   * of the appropriate size.
   */
  void setNext(const int nupdates = 1);

  /**
   * Get buffer to manually set random variates.
   *
   * @return Buffer. Rows index trajectories, columns time (outer) and
   * variable (inner).
   */
  matrix_type& buf();

private:
  /**
   * Random number generator.
   */
  Random& rng;

  /**
   * Number of updates left to skip.
   */
  int nskip;

  /**
   * Number of updates set.
   */
  int nset;

  /**
   * Buffer.
   */
  matrix_type X;

  /* net sizes, for convenience */
  static const int NR = net_size<B,typename B::RTypeList>::value;
};
}

#include "RUpdateVisitor.hpp"
#include "../traits/random_traits.hpp"
#include "../traits/derived_traits.hpp"

template<class B, bi::Location CL>
bi::RUpdater<B,CL>::RUpdater(Random& rng) : rng(rng), nskip(0), nset(0) {
  //
}

template<class B, bi::Location CL>
template<bi::Location L>
void bi::RUpdater<B,CL>::update(const real t, const real tnxt, State<L>& s) {
  typedef typename B::RTypeList S;
  typedef typename State<L>::matrix_reference_type M1;
  typedef RUpdateVisitor<B,S,M1> Visitor;

  if (nskip == 0) {
    BOOST_AUTO(Y, s.get(R_NODE));
    if (nset > 0) {
      /* retrieve from buffer */
      BI_ASSERT(Y.size1() == X.size1(),
          "Incompatible sizes for r-node buffer update");
      Y = columns(X, X.size2() - NR*nset, NR);
      --nset;
    } else {
      /* generate from rng */
      if (Y.lead() == Y.size1()) {
        if (all_gaussian_variates<S>::value) {
          rng.gaussians(vec(Y));
        } else if (all_uniform_variates<S>::value) {
          rng.uniforms(vec(Y), -0.5, 0.5);
        } else if (all_wiener_increments<S>::value) {
          if (fabs(tnxt - t) > 0.0) {
            rng.gaussians(vec(Y), 0.0, std::sqrt(fabs(tnxt - t)));
          } else {
            Y.clear();
          }
        } else {
          Visitor::accept(rng, t, tnxt, Y);
        }
      } else {
        Visitor::accept(rng, t, tnxt, Y);
      }
    }
  } else {
    /* skip */
    --nskip;
  }

  /* post-condition */
  assert (nskip >= 0);
  assert (nset >= 0);
}

template<class B, bi::Location CL>
inline void bi::RUpdater<B,CL>::skipNext(const int nupdates) {
  /* pre-condition */
  BI_WARN(nskip == 0, "Previous skips have not been exhausted");
  BI_WARN(nset == 0, "Previous buffer has not been exhausted");

  nskip = nupdates;
}

template<class B, bi::Location CL>
inline void bi::RUpdater<B,CL>::setNext(const int nupdates) {
  typedef typename B::RTypeList S;

  /* pre-condition */
  assert(X.size2() == nupdates*NR);
  BI_WARN(nskip == 0, "Previous skips have not been exhausted");
  BI_WARN(nset == 0, "Previous buffer has not been exhausted");

  nset = nupdates;
}

template<class B, bi::Location CL>
inline typename bi::RUpdater<B,CL>::matrix_type& bi::RUpdater<B,CL>::buf() {
  return X;
}

#endif
