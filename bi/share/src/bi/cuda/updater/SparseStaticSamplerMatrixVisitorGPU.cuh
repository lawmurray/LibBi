/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICSAMPLERMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_SPARSESTATICSAMPLERMATRIXVISITORGPU_CUH

#include "../random/RngGPU.cuh"

namespace bi {
/**
 * Matrix visitor for sparse static sampling, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class SparseStaticSamplerMatrixVisitorGPU {
public:
  /**
   * Update.
   *
   * @param[in,out] rng Random number generator.
   * @param s State.
   * @param mask Mask.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, State<B,ON_DEVICE>& s,
      const Mask<ON_DEVICE>& mask, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of SparseStaticSamplerMatrixVisitorGPU.
 */
template<class B, class PX, class OX>
class SparseStaticSamplerMatrixVisitorGPU<B,empty_typelist,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, State<B,ON_DEVICE>& s,
      const Mask<ON_DEVICE>& mask, const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
inline void bi::SparseStaticSamplerMatrixVisitorGPU<B,S,PX,OX>::accept(
    RngGPU& rng, State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
    const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;

  const int Q = gridDim.x * blockDim.x;  // number of threads
  const int id = var_id<target_type>::value;
  int p;

  if (mask.isDense(id)) {
    p = blockIdx.x*blockDim.x + threadIdx.x;
    while (p < s.size()) {
      front::samples(rng, s, p, pax, x);
      p += Q;
    }
  } else if (mask.isSparse(id)) {
    BI_ASSERT_MSG(false, "Cannot do sparse update with matrix expression");
  }
  SparseStaticSamplerMatrixVisitorGPU<B,pop_front,PX,OX>::accept(rng, s, mask,
      pax, x);
}

#endif
