/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1466 $
 * $Date: 2011-05-09 17:37:34 +0800 (Mon, 09 May 2011) $
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
   * @param N Number of variables in unscented transformation.
   * @param a Scaling factor for variates.
   */
  void prepare(const int N, const real a);

private:
  /**
   * Number of variables in unscented transformation.
   */
  int N;

  /**
   * Scaling factor for variates for upcoming updates.
   */
  real a;

  static const int ND = net_size<B,typename B::DTypeList>::value;
  static const int NC = net_size<B,typename B::CTypeList>::value;
  static const int NR = net_size<B,typename B::RTypeList>::value;
  static const int NP = net_size<B,typename B::PTypeList>::value;
  static const int M = ND + NC + NR + ((SH == STATIC_OWN) ? NP : 0);
};
}

#include "UnscentedORUpdateVisitor.hpp"
#include "../traits/likelihood_traits.hpp"
#include "../traits/derived_traits.hpp"

template<class B, bi::StaticHandling SH>
bi::UnscentedORUpdater<B,SH>::UnscentedORUpdater() : N(0), a(0.0) {
  //
}

template<class B, bi::StaticHandling SH>
void bi::UnscentedORUpdater<B,SH>::prepare(const int N, const real a) {
  this->N = N;
  this->a = a;
}

template<class B, bi::StaticHandling SH>
template<bi::Location L>
void bi::UnscentedORUpdater<B,SH>::update(const SparseMask<L>& mask,
    State<L>& s) {
  /* pre-condition */
  assert (L == ON_HOST);

  typedef typename B::OTypeList S;
  typedef typename State<L>::vector_reference_type V1;
  typedef UnscentedORUpdateVisitor<B,S,V1> Visitor;

  int i, size;
  int W = mask.size(), start = 1 + M;

  if (W > 0) {
    BOOST_AUTO(d1, diagonal(rows(s.get(OR_NODE), start, W)));
    BOOST_AUTO(d2, diagonal(rows(s.get(OR_NODE), start + N, W)));

    if (all_gaussian_likelihoods<S>::value || all_log_normal_likelihoods<S>::value) {
      bi::fill(d1.begin(), d1.end(), a);
      bi::fill(d2.begin(), d2.end(), -a);
    } else {
      start = 0;

      /* dense mask blocks */
      BOOST_AUTO(iter1, mask.getDenseMask().begin());
      while (iter1 != mask.getDenseMask().end()) {
        BOOST_AUTO(ids, iter1->getIds());
        size = iter1->size()/ids.size();
        for (i = 0; i < ids.size(); ++i) {
          Visitor::accept(ids[i], a, subrange(d1, start, size),
              subrange(d2, start, size));
          start += size;
        }
        ++iter1;
      }

      /* sparse mask blocks */
      BOOST_AUTO(iter2, mask.getSparseMask().begin());
      while (iter2 != mask.getSparseMask().end()) {
        BOOST_AUTO(ids, iter2->getIds());
        size = iter2->size()/ids.size();
        for (i = 0; i < ids.size(); ++i) {
          Visitor::accept(ids[i], a, subrange(d1, start, size),
              subrange(d2, start, size));
          start += size;
        }
        ++iter2;
      }
    }
  }
}

#endif
