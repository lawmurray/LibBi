/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_DUPDATER_HPP
#define BI_UPDATER_DUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for d-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 */
template<class B, StaticHandling SH = STATIC_SHARED>
class DUpdater {
public:
  /**
   * Update d-net.
   *
   * @param t Start of interval (open).
   * @param tnxt End of interval (closed).
   * @param s State to update.
   */
  void update(const real t, const real tnxt, State<ON_HOST>& s);

  /**
   * @copydoc update(const real, const real, State<ON_HOST>&)
   */
  void update(const real t, const real tnxt, State<ON_DEVICE>& s);
};

}

#include "DUpdateVisitor.hpp"
#include "../state/Pa.hpp"
#include "../host/bind.hpp"
#ifdef USE_SSE
#include "../host/sse_host.hpp"
#include "../host/sse_const_host.hpp"
#endif

template<class B, bi::StaticHandling SH>
void bi::DUpdater<B,SH>::update(const real t, const real tnxt,
    State<ON_HOST>& s) {
  #ifdef USE_SSE
  typedef typename B::DTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,sse_const_host,sse_host>::type pa;
  typedef Pa<B,sse_real,pa,sse_host,sse_host,pa,sse_host,sse_host,sse_host> V1;
  typedef sse_real V2;
  typedef DUpdateVisitor<ON_HOST,B,S,V1,V2> Visitor;

  static const int N = net_size<B,S>::value;
  const int P = s.size();

  if (N > 0) {
    bind(s);
    #pragma omp parallel
    {
      int p, id;
      V1 pax(0);
      V2 xnxt[N];

      #pragma omp for
      for (p = 0; p < P; p += BI_SSE_SIZE) {
        pax.p = p;
        Visitor::accept(t, pax, tnxt, xnxt);
        for (id = 0; id < N; ++id) {
          sse_state_set(s.get(D_NODE), pax.p, id, xnxt[id]);
        }
      }
    }
    unbind(s);
  }
  #else
  typedef typename B::DTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<B,real,pa,host,host,pa,host,host,host> V1;
  typedef real V2;
  typedef DUpdateVisitor<ON_HOST,B,S,V1,V2> Visitor;

  static const int N = net_size<B,S>::value;
  const int P = s.size();

  if (N > 0) {
    bind(s);
    #pragma omp parallel
    {
      int p, id;
      V1 pax(0);
      V2 xnxt[N];

      #pragma omp for
      for (p = 0; p < P; ++p) {
        pax.p = p;
        Visitor::accept(t, pax, tnxt, xnxt);
        for (id = 0; id < N; ++id) {
          s.get(D_NODE)(pax.p, id) = xnxt[id];
        }
      }
    }
    unbind(s);
  }
  #endif
}

#ifdef __CUDACC__
#include "../cuda/updater/DUpdater.cuh"
#endif

#endif
