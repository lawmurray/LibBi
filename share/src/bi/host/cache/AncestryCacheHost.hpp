/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_CACHE_ANCESTRYCACHEHOST_HPP
#define BI_HOST_CACHE_ANCESTRYCACHEHOST_HPP

namespace bi {
class AncestryCacheHost {
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
  static int insert(M1& X, V1& as, V1& os, V1& ls, const int start, const M2 X1,
      const V2 as1);
};
}

#include "../../math/temp_vector.hpp"
#include "../../math/temp_matrix.hpp"
#include "../../math/view.hpp"
#include "../../primitive/vector_primitive.hpp"
#include "../../primitive/matrix_primitive.hpp"

template<class V1>
int bi::AncestryCacheHost::prune(V1& as, V1& os, V1& ls) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);

  int i, j, numRemoved = 0;
  for (i = 0; i < ls.size(); ++i) {
    j = ls(i);
    while (os(j) == 0) {
      ++numRemoved;
      j = as(j);
      if (j >= 0) {
        --os(j);
      } else {
        break;
      }
    }
  }
  return numRemoved;
}

template<class M1, class V1, class M2, class V2>
int bi::AncestryCacheHost::insert(M1& X, V1& as, V1& os, V1& ls, const int start,
    const M2 X1, const V2 as1) {
  /* pre-condition */
  BI_ASSERT(X1.size1() == as1.size());
  BI_ASSERT(!M1::on_device);
  BI_ASSERT(!V1::on_device);

  typedef typename temp_host_vector<int>::type host_int_vector_type;

  const int N = X1.size1();
  host_int_vector_type bs(N);
  int i, q = start;

  bi::gather(as1, ls, bs);
  ls.resize(N, false);

  for (i = 0; i < N; ++i) {
    while (os(q) > 0) {
      ++q;
      if (q == X.size1()) {
        q = 0;
      }
    }
    ls(i) = q;
    ++q;
    if (q == X.size1()) {
      q = 0;
    }
  }

  bi::scatter(ls, bs, as);
  bi::scatter_rows(ls, X1, X);

  return q;
}

#endif
