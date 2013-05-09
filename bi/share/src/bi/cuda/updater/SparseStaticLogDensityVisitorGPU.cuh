/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYVISITORGPU_CUH
#define BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYVISITORGPU_CUH

namespace bi {
/**
 * Visitor for sparse static log-density updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class SparseStaticLogDensityVisitorGPU {
public:
  /**
   * Update log-density.
   *
   * @tparam T1 Scalar type.
   *
   * @param s State.
   * @param mask Mask.
   * @param p Trajectory id.
   * @param pax Parents.
   * @param[out] x Output.
   * @param[in,out] lp Log-density.
   */
  template<class T1>
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s,
      const Mask<ON_DEVICE>& mask, const int p, const PX& pax, OX& x,
      T1& lp);
};

/**
 * @internal
 *
 * Base case of SparseStaticLogDensityVisitorGPU.
 */
template<class B, class PX, class OX>
class SparseStaticLogDensityVisitorGPU<B,empty_typelist,PX,OX> {
public:
  template<class T1>
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s,
      const Mask<ON_DEVICE>& mask, const int p, const PX& pax, OX& x,
      T1& lp) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
template<class T1>
inline void bi::SparseStaticLogDensityVisitorGPU<B,S,PX,OX>::accept(
    State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask, const int p,
    const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename front::coord_type coord_type;

  const int id = var_id<target_type>::value;
  const int size = mask.getSize(id);
  int i, ix;
  coord_type cox;

  for (i = 0; i < size; ++i) {
    ix = mask.getIndex(id, i);
    cox.setIndex(ix);
    front::logDensities(s, p, ix, cox, pax, x, lp);
  }

  SparseStaticLogDensityVisitorGPU<B,pop_front,PX,OX>::accept(s, mask, p, pax,
      x, lp);
}

#endif
