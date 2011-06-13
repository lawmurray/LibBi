/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_OUPDATER_HPP
#define BI_UPDATER_OUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../method/misc.hpp"
#include "../buffer/SparseMask.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for o-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH = STATIC_SHARED>
class OUpdater {
public:
  /**
   * Update o-net.
   *
   * @tparam L Location.
   *
   * @param mask Mask.
   * @param s State to update.
   */
  template<Location L>
  void update(const SparseMask<L>& mask, State<ON_HOST>& s);

  /**
   * @copydoc update(const SparseMask<L>& mask, State<ON_HOST>& s)
   */
  template<Location L>
  void update(const SparseMask<L>& mask, State<ON_DEVICE>& s);
};
}

#include "OUpdateVisitor.hpp"
#include "../state/Pa.hpp"
#include "../host/host.hpp"
#include "../host/const_host.hpp"

template<class B, bi::StaticHandling SH>
template<bi::Location L>
void bi::OUpdater<B,SH>::update(const SparseMask<L>& mask, State<ON_HOST>& s) {
  typedef typename B::OTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<ON_HOST,B,real,pa,host,host,pa,host,host,host> V3;
  typedef OUpdateVisitor<B,S,V3,real,real> Visitor;

  const int P = s.size();
  const int W = mask.size();

  if (W > 0) {
    s.oresize(W, false);
    bind(s);
    #pragma omp parallel
    {
      int j = 0, i, id, p;
      Coord cox;
      V3 pax(0);
      real r, o;

      BOOST_AUTO(iter1, mask.getDenseMask().begin());
      while (iter1 != mask.getDenseMask().end()) {
        for (i = 0; i < (*iter1)->size(); ++i, ++j) {
          (*iter1)->coord(i, id, cox);

          #pragma omp for
          for (p = 0; p < P; ++p) {
            pax.p = p;
            r = hostORState(pax.p, j);
            Visitor::accept(id, cox, pax, r, o);
            hostOState(pax.p, j) = o;
          }
        }
        ++iter1;
      }

      BOOST_AUTO(iter2, mask.getSparseMask().begin());
      while (iter2 != mask.getSparseMask().end()) {
        for (i = 0; i < (*iter2)->size(); ++i, ++j) {
          (*iter2)->coord(i, id, cox);

          #pragma omp for
          for (p = 0; p < P; ++p) {
            pax.p = p;
            r = hostORState(pax.p, j);
            Visitor::accept(id, cox, pax, r, o);
            hostOState(pax.p, j) = o;
          }
        }
        ++iter2;
      }

      assert (j == mask.size());
    }
    unbind(s);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/OUpdater.cuh"
#endif

#endif
