/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_UNSCENTEDORUPDATER_HPP
#define BI_UPDATER_UNSCENTEDORUPDATER_HPP

#include "../state/State.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * Specialised updater for or-net, for use with UnscentedKalmanFilter.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH>
class UnscentedORUpdater {
public:
  /**
   * Constructor.
   */
  UnscentedORUpdater();

  /**
   * Update r-net.
   *
   * @tparam L Location.
   *
   * @param mask Mask.
   * @param s State to update.
   */
  template<Location L>
  void update(const SparseMask<L>& mask, State<L>& s);

  /**
   * Prepare for upcoming update.
   *
   * @param N1 Number of unconditioned variables in unscented transformation.
   * @param W Number of observations.
   * @param a Scaling factor for variates.
   * @param fixed True if starting state fixed, false otherwise.
   */
  void prepare(const int N1, const int W, const real a,
      const bool fixed = false);

private:
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
};
}

#include "UnscentedORUpdateVisitor.hpp"
#include "../traits/likelihood_traits.hpp"
#include "../traits/derived_traits.hpp"

template<class B, bi::StaticHandling SH>
bi::UnscentedORUpdater<B,SH>::UnscentedORUpdater() : N1(0), W(0), a(0.0),
    fixed(false) {
  //
}

template<class B, bi::StaticHandling SH>
void bi::UnscentedORUpdater<B,SH>::prepare(const int N1, const int W,
    const real a, const bool fixed) {
  this->N1 = N1;
  this->W = W;
  this->a = a;
  this->fixed = fixed;
}

template<class B, bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedORUpdater<B,SH>::update(const SparseMask<L>& mask,
    State<L>& s) {
  /* pre-condition */
  assert (mask.size() == W);

  typedef typename B::OTypeList S;
  typedef typename State<L>::vector_reference_type V1;
  typedef UnscentedORUpdateVisitor<B,S,V1> Visitor;

  const int P = 2*N1 + 1;
  int i, p, size, start = 1 + N1 - W;

  if (W > 0) {
    s.get(OR_NODE).clear();
    for (p = 0; p < s.size(); p += P) {
      BOOST_AUTO(d1, diagonal(rows(s.get(OR_NODE), p + start, W)));
      BOOST_AUTO(d2, diagonal(rows(s.get(OR_NODE), p + start + N1, W)));

      if (all_gaussian_likelihoods<S>::value || all_log_normal_likelihoods<S>::value) {
        bi::fill(d1.begin(), d1.end(), a);
        bi::fill(d2.begin(), d2.end(), -a);
      } else {
        start = 0;

        /* dense mask blocks */
        BOOST_AUTO(iter1, mask.getDenseMask().begin());
        BOOST_AUTO(end1, mask.getDenseMask().end());
        while (iter1 != end1) {
          BOOST_AUTO(ids, (*iter1)->getIds());
          size = (*iter1)->size()/ids.size();
          for (i = 0; i < ids.size(); ++i) {
            Visitor::accept(*(ids.begin() + i), a, subrange(d1, start, size),
                subrange(d2, start, size));
            start += size;
          }
          ++iter1;
        }

        /* sparse mask blocks */
        BOOST_AUTO(iter2, mask.getSparseMask().begin());
        BOOST_AUTO(end2, mask.getSparseMask().end());
        while (iter2 != end2) {
          BOOST_AUTO(ids, (*iter2)->getIds());
          size = (*iter2)->size()/ids.size();
          for (i = 0; i < ids.size(); ++i) {
            Visitor::accept(*(ids.begin() + i), a, subrange(d1, start, size),
                subrange(d2, start, size));
            start += size;
          }
          ++iter2;
        }
      }
    }
  }
}

#endif
