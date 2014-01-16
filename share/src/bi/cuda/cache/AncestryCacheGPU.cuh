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
  static int prune(V1& as, V1& os, V1& ls);

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
  static int insert(M1& X, V1& as, V1& os, V1& ls, const int start, const M2 X1, const V2 as1);
};
}

#include "AncestryCacheKernel.cuh"
#include "../../math/temp_vector.hpp"
#include "../../math/temp_matrix.hpp"
#include "../../math/view.hpp"
#include "../../primitive/vector_primitive.hpp"
#include "../../primitive/matrix_primitive.hpp"

template<class V1>
int bi::AncestryCacheGPU::prune(V1& as, V1& os, V1& ls) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);

  typename temp_gpu_vector<int>::type numRemoved(ls.size());

  const int N = ls.size();
  dim3 Db, Dg;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), N);
  Dg.x = (N + Db.x - 1)/Db.x;

  kernelAncestryCachePrune<<<Dg,Db>>>(as, os, ls, numRemoved);
  CUDA_CHECK;

  return sum_reduce(numRemoved);
}

template<class M1, class V1, class M2, class V2>
int bi::AncestryCacheGPU::insert(M1& X, V1& as, V1& os, V1& ls, const int start, const M2 X1,
    const V2 as1) {
  /* pre-condition */
  BI_ASSERT(!M1::on_device);
  BI_ASSERT(V1::on_device);
  BI_ASSERT(M2::on_device);
  BI_ASSERT(V2::on_device);
  BI_ASSERT(X1.size1() == as1.size());

  const int N = X1.size1();
  typename temp_gpu_vector<int>::type Z(os.size()), bs(N);

  BOOST_AUTO(seq, thrust::make_counting_iterator(0));

  bi::gather(as1, ls, bs);
  ls.resize(N, false);

  /* first fit */
  zero_inclusive_scan(os, Z);
  thrust::upper_bound(Z.fast_begin(), Z.fast_end(), seq, seq + N, ls.fast_begin());
  int q = 0;

  /* next fit */
//  int q = start, len, numAlloc, maxAlloc, numDone = 0;
//  do {
//    /* determine subrange to search */
//    len = bi::min(Z.size(), os.size() - q);
//
//    /* count up free slots in this subrange */
//    BOOST_AUTO(z, subrange(Z, 0, len));
//    zero_inclusive_scan(subrange(os, q, len), z);
//
//    /* number of free slots to allocate in this subrange */
//    maxAlloc = *(z.end() - 1);
//    numAlloc = bi::min(maxAlloc, N - numDone);
//
//    /* allocate slots */
//    thrust::upper_bound(z.fast_begin(), z.fast_end(), seq, seq + numAlloc, ls.fast_begin() + numDone);
//    addscal_elements(subrange(ls, numDone, numAlloc), q, subrange(ls, numDone, numAlloc));
//
//    numDone += numAlloc;
//    q += z.size();
//    if (q >= os.size()) {
//      q = 0;
//    }
//  } while (numDone < N);

  bi::scatter(ls, bs, as);
  bi::scatter_rows(ls, X1, X);

  return q;
}

#endif
