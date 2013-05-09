/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICUPDATERVISITORGPU_CUH
#define BI_CUDA_UPDATER_SPARSESTATICUPDATERVISITORGPU_CUH

namespace bi {
/**
 * Visitor for sparse static updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class SparseStaticUpdaterVisitorGPU {
public:
  /**
   * Update.
   *
   * @param s State.
   * @param mask Mask.
   * @param p Trajectory id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
      const int p, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of SparseStaticUpdaterVisitorGPU.
 */
template<class B, class PX, class OX>
class SparseStaticUpdaterVisitorGPU<B,empty_typelist,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
      const int p, const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
inline void bi::SparseStaticUpdaterVisitorGPU<B,S,PX,OX>::accept(
    State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask, const int p, const PX& pax, OX& x) {
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
    front::simulates(s, p, ix, cox, pax, x);
  }

  SparseStaticUpdaterVisitorGPU<B,pop_front,PX,OX>::accept(s, mask, p, pax, x);
}

#endif
