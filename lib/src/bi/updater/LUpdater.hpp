/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_LUPDATER_HPP
#define BI_UPDATER_LUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../buffer/SparseMask.hpp"

namespace bi {
/**
 * Calculator for likelihood of states given observations.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 */
template<class B, bi::StaticHandling SH>
class LUpdater {
public:
  /**
   * Update log-likelihood weights.
   *
   * @tparam V2 Vector type.
   *
   * @param s State.
   * @param mask Sparse mask for observations in @p s.
   * @param[in,out] lws Log-weights.
   */
  template<class V2>
  void update(State<ON_HOST>& s, const SparseMask<ON_HOST>& mask, V2& lws);

  /**
   * @copydoc update(State<ON_HOST>&, const V1&, V2&)
   */
  template<class V2>
  void update(State<ON_DEVICE>& s, const SparseMask<ON_DEVICE>& mask, V2& lws);
};
}

#include "LUpdateVisitor.hpp"
#include "../state/Pa.hpp"

template<class B, bi::StaticHandling SH>
template<class V2>
void bi::LUpdater<B,SH>::update(State<ON_HOST>& s, const SparseMask<ON_HOST>& mask,
    V2& lws) {
  typedef typename B::OTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<ON_HOST,B,real,pa,host,host,pa,host,host,host> V3;
  typedef LUpdateVisitor<B,S,V3,real,real> Visitor;

  const int P = lws.size();
  if (mask.size() > 0) {
    bind(s);
    #pragma omp parallel
    {
      int i, j = 0, id, p;

      Coord cox;
      V3 pax(0);
      real y, l = 0.0;

      BOOST_AUTO(iter1, mask.getDenseMask().begin());
      BOOST_AUTO(end1, mask.getDenseMask().end());
      while (iter1 != end1) {
        for (i = 0; i < (*iter1)->size(); ++i, ++j) {
          (*iter1)->coord(i, id, cox);
          y = hostOYState(0, j);

          #pragma omp for
          for (p = 0; p < P; ++p) {
            pax.p = p;
            Visitor::accept(id, cox, pax, y, l);
            lws(p) += l;
          }
        }
        ++iter1;
      }

      BOOST_AUTO(iter2, mask.getSparseMask().begin());
      BOOST_AUTO(end2, mask.getSparseMask().end());
      while (iter2 != end2) {
        for (i = 0; i < (*iter2)->size(); ++i, ++j) {
          (*iter2)->coord(i, id, cox);
          y = hostOYState(0, j);

          #pragma omp for
          for (p = 0; p < P; ++p) {
            pax.p = p;
            Visitor::accept(id, cox, pax, y, l);
            lws(p) += l;
          }
        }
        ++iter2;
      }
    }
    unbind(s);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/LUpdater.cuh"
#endif

#endif
