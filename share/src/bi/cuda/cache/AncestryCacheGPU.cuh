/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_CACHE_ANCESTRYCACHEGPU_CUH
#define BI_CUDA_CACHE_ANCESTRYCACHEGPU_CUH

namespace bi {
class AncestryCacheGPU {
public:
  /**
   * Prune ancestry tree.
   *
   * @tparam V1 Integer vector type.
   *
   * @param as Ancestors.
   * @param os Offspring.
   * @param ls Leaves.
   *
   * @return Number of nodes removed.
   */
  template<class V1>
  static int prune(V1 as, V1 os, V1 ls);

  /**
   * Insert into ancestry tree.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Integer vector type.
   * @tparam M2 Matrix type.
   * @tparam V2 Integer vector type.
   *
   * @param X Particle storage.
   * @param as Ancestry storage.
   * @param os Offspring storage.
   * @param ls Leaves storage.
   * @param start Starting index into storage for search.
   * @param X1 Particles to insert.
   * @param as1 Ancestry to insert.
   *
   * @return Updated starting index into storage.
   */
  template<class M1, class V1, class M2, class V2>
  static int insert(M1 X, V1 as, V1 os, V1 ls, const int start, const M2 X1, const V2 as1);
};
}

#include "AncestryCacheKernel.cuh"
#include "../../math/temp_vector.hpp"
#include "../../math/temp_matrix.hpp"
#include "../../math/view.hpp"
#include "../../primitive/vector_primitive.hpp"
#include "../../primitive/matrix_primitive.hpp"

template<class V1>
int bi::AncestryCacheGPU::prune(V1 as, V1 os, V1 ls) {
  /* pre-condition */
  assert(!V1::on_device);

  typename temp_gpu_vector<int>::type numRemoved(ls.size());

  const int N = ls.size();
  dim3 Db, Dg;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), N);
  Dg.x = (N + Db.x - 1)/Db.x;

  kernelAncestryCachePrune<V1,BOOST_TYPEOF(numRemoved)><<<Dg,Db>>>(as, os, ls, numRemoved);
  CUDA_CHECK;

  return sum_reduce(numRemoved);
}

template<class M1, class V1, class M2, class V2>
int bi::AncestryCacheGPU::insert(M1 X, V1 as, V1 os, V1 ls, const int start, const M2 X1,
    const V2 as1) {
  /* pre-condition */
  BI_ASSERT(X1.size1() == as1.size());

  const int N = X1.size1();
  typename temp_gpu_vector<int>::type Z(os.size()), bs(N), seq(N);

  seq_elements(seq, 0);
  bi::gather(as1, ls, bs);
  ls.resize(N, false);
  zero_inclusive_scan(os, Z);
  bi::upper_bound(Z, seq, ls);
  bi::scatter(ls, bs, as);
  bi::scatter_rows(ls, X1, X);

  return 0;
}

#endif
