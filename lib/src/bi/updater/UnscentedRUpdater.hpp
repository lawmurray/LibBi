/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_UNSCENTEDRUPDATER_HPP
#define BI_UPDATER_UNSCENTEDRUPDATER_HPP

#include "../state/State.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * Specialised updater for r-net, for use with UnscentedKalmanFilter.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH>
class UnscentedRUpdater {
public:
  /**
   * Constructor.
   */
  UnscentedRUpdater();

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
   * Prepare for upcoming update.
   *
   * @param nsteps Number of updates to be performed.
   * @param N1 Number of unconditioned variables in unscented transformation.
   * @param W Number of observations.
   * @param a Scaling factor for variates.
   * @param fixed True if starting state fixed, false otherwise.
   */
  void prepare(const int nsteps, const int N1, const int W, const real a,
      const bool fixed = false);

private:
  /**
   * Number of steps captured by state for upcoming updates.
   */
  int nsteps;

  /**
   * Number of unconditioned variables in unscented transformation.
   */
  int N1;

  /**
   * Number of observations.
   */
  int W;

  /**
   * Scaling factor for variates for upcoming updates.
   */
  real a;

  /**
   * Is starting state fixed for this transformation?
   */
  bool fixed;

  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
};
}

#include "UnscentedRUpdateVisitor.hpp"
#include "../traits/random_traits.hpp"
#include "../traits/derived_traits.hpp"

template<class B, bi::StaticHandling SH>
bi::UnscentedRUpdater<B,SH>::UnscentedRUpdater() : nsteps(0), N1(0), W(0),
    a(0.0) {
  //
}

template<class B, bi::StaticHandling SH>
void bi::UnscentedRUpdater<B,SH>::prepare(const int nsteps, const int N1,
    const int W, const real a, const bool fixed) {
  /* pre-condition */
  BI_ASSERT(this->nsteps == 0,
      "Simulation for previous unscented transformation incomplete");

  this->nsteps = nsteps;
  this->N1 = N1;
  this->W = W;
  this->a = a;
  this->fixed = fixed;
}

template<class B, bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedRUpdater<B,SH>::update(const real t, const real tnxt,
    State<L>& s) {
  /* pre-conditions */
  BI_ASSERT(this->nsteps > 0, "Updates for random variates of " <<
      "unscented transformation have been exhausted");
  assert (s.size() % (2*N1 + 1) == 0);

  typedef typename B::RTypeList S;
  typedef typename State<L>::vector_reference_type V1;
  typedef UnscentedRUpdateVisitor<B,S,V1> Visitor;

  const int P = 2*N1 + 1;
  int start, p;

  if (nsteps == 1) {
    /* last step uses primary block */
    start = 1 + (fixed ? 0 : ND + NC);
  } else {
    /* other steps use extra blocks */
    start = 1 + N1 - W - (nsteps - 1)*NR;
  }

  s.get(R_NODE).clear();
  for (p = 0; p < s.size(); p += P) {
    BOOST_AUTO(d1, diagonal(rows(s.get(R_NODE), p + start, NR)));
    BOOST_AUTO(d2, diagonal(rows(s.get(R_NODE), p + start + N1, NR)));

    if (all_gaussian_variates<S>::value) {
      bi::fill(d1.begin(), d1.end(), a);
      bi::fill(d2.begin(), d2.end(), -a);
    } else if (all_uniform_variates<S>::value) {
      bi::fill(d1.begin(), d1.end(), a*std::sqrt(1.0/12.0));
      bi::fill(d2.begin(), d2.end(), -a*std::sqrt(1.0/12.0));
    } else if (all_wiener_increments<S>::value) {
      if (tnxt - t > 0.0) {
        bi::fill(d1.begin(), d1.end(), a*std::sqrt(tnxt - t));
        bi::fill(d2.begin(), d2.end(), -a*std::sqrt(tnxt - t));
      } else {
        d1.clear();
        d2.clear();
      }
    } else {
      Visitor::accept(t, tnxt, a, d1, d2);
    }
  }
  --nsteps;
}

#endif
