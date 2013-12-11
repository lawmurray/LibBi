/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_MULTIOPERATION_HPP
#define BI_MATH_MULTIOPERATION_HPP

#include "operation.hpp"

namespace bi {
/**
 * @name Getters and setters
 */
//@{
/**
 * Get single matrix from a multi-matrix.
 *
 * @ingroup math_multi_op
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param P Number of matrices in the multi-matrix.
 * @param Xs The multi-matrix.
 * @param p Index of the matrix.
 * @param[out] X The matrix.
 */
template<class M1, class M2>
void multi_get_matrix(const int P, const M1 Xs, const int p, M2 X);

/**
 * Set single matrix in a multi-matrix.
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param P Number of matrices in the multi-matrix.
 * @param[in,out] Xs The multi-matrix.
 * @param p Index of the matrix.
 * @param X The matrix.
 */
template<class M1, class M2>
void multi_set_matrix(const int P, M1 Xs, const int p, const M2 X);

/**
 * Get single vector from a multi-vector.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param P Number of vectors in the multi-vector.
 * @param xs The multi-vector.
 * @param p Index of the vector.
 * @param[out] x The vector.
 */
template<class V1, class V2>
void multi_get_vector(const int P, const V1 xs, const int p, V2 x);

/**
 * Set single vector in a multi-vector.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param P Number of vectors in the multi-vector.
 * @param[in,out] xs The multi-vector.
 * @param p Index of the vector.
 * @param x The vector.
 */
template<class V1, class V2>
void multi_set_vector(const int P, V1 xs, const int p, const V2 x);
//@}

/**
 * Multiple #transpose.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_transpose(const int P, const M1 A, M2 B);

/**
 * Multiple #chol.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_chol(const int P, const M1 A, M2 U, char uplo = 'U',
    const CholeskyStrategy = ADJUST_DIAGONAL) throw (CholeskyException);

/**
 * Multiple #matrix_axpy.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_matrix_axpy(const int P, const typename M1::value_type a,
    const M1 X, M2 Y, const bool clear = false);

/**
 * Multiple #matrix_scal.
 *
 * @ingroup math_multi_op
 */
template<class M1>
void multi_matrix_scal(const int P, typename M1::value_type alpha, M1 X);

/**
 * Multiple #condition.
 *
 * @ingroup math_multi_op
 */
template<class V1, class M1, class V2, class M2, class M3, class V3>
void multi_condition(const int P, V1 mu1, M1 U1, const V2 mu2, const M2 U2,
    const M3 C, const V3 x2);

/**
 * @name QRUpdate (and QRUpdate-like) adapters
 */
//@{
/**
 * Multiple #chkup.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2, class V2>
void multi_chkup(const int P, M1 U, M2 A, V2 b);

/**
 * Multiple #ch1up.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1, class V2>
void multi_ch1up(const int P, M1 U, V1 a, V2 b);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_ch1up_impl {
  template<class M1, class V1, class V2>
  static void func(const int P, M1 U, V1 a, V2 b);
};

/**
 * Multiple #chkdn.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2, class V2>
void multi_chkdn(const int P, M1 U, M2 A, V2 b) throw (CholeskyException);

/**
 * Multiple #ch1dn.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1, class V2>
void multi_ch1dn(const int P, M1 U, V1 a, V2 b) throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_ch1dn_impl {
  template<class M1, class V1, class V2>
  static void func(const int P, M1 U, V1 a, V2 b) throw (CholeskyException);
};

//@}
/**
 * @name BLAS (and BLAS-like) adapters
 */
//@{
/**
 * Multiple #scal.
 *
 * @ingroup math_multi_op
 */
template<class V1>
void multi_scal(const int P, typename V1::value_type alpha, V1 x);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_scal_impl {
  template<class V1>
  static void func(const int P, typename V1::value_type alpha, V1 x);
};

/**
 * @internal
 */
template<Location L, class T1>
struct multi_dot_impl {
  template<class V1, class V2>
  static typename V1::value_type func(const int P, const V1 a, const V2 b);

  template<class V1>
  static typename V1::value_type func(const int P, const V1 a);
};

/**
 * Multiple #multi_axpy.
 *
 * @ingroup math_multi_op
 */
template<class V1, class V2>
void multi_axpy(const int P, const typename V1::value_type a, const V1 x, V2 y,
    const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_axpy_impl {
  template<class V1, class V2>
  static void func(const int P, const typename V1::value_type a, const V1 x, V2 y,
      const bool clear = false);
};

/**
 * Multiple #gemv.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1, class V2>
void multi_gemv(const int P, const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_gemv_impl {
  template<class M1, class V1, class V2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A,
      const V1 x, const typename V2::value_type beta, V2 y,
      const char transA);
};

/**
 * Multiple #symv.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1, class V2>
void multi_symv(const int P, const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char uplo = 'U');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_symv_impl {
  template<class M1, class V1, class V2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, const V1 x,
      const typename V2::value_type beta, V2 y, const char uplo);
};

/**
 * Multiple #trmv.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1>
void multi_trmv(const int P, const M1 A, V1 x, const char uplo = 'U', const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_trmv_impl {
  template<class M1, class V1>
  static void func(const int P, const M1 A, V1 x, const char uplo, const char transA);
};

/**
 * Multiple #gdmv.
 *
 * @ingroup math_multi_op
 */
template<class V1, class V2, class V3>
void multi_gdmv(const int P, const typename V1::value_type alpha, const V1 A, const V2 x,
    const typename V3::value_type beta, V3 y);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_gdmv_impl {
  template<class V1, class V2, class V3>
  static void func(const int P, const typename V1::value_type alpha, const V1 A, const V2 x,
      const typename V3::value_type beta, V3 y);
};

/**
 * Multiple #gemm.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2, class M3>
void multi_gemm(const int P, const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char transA = 'N',
    const char transX = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_gemm_impl {
  template<class M1, class M2, class M3>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, const M2 X,
      const typename M3::value_type beta, M3 Y, const char transA,
      const char transX);
};

/**
 * Multiple #symm.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2, class M3>
void multi_symm(const int P, const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char side = 'L',
    const char uplo = 'U');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_symm_impl {
  template<class M1, class M2, class M3>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, const M2 X,
      const typename M3::value_type beta, M3 Y, const char side,
      const char uplo);
};

/**
 * Multiple #trmm.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_trmm(const int P, const typename M1::value_type alpha, const M1 A, M2 B,
    const char side = 'L', const char uplo = 'U', const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_trmm_impl {
  template<class M1, class M2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, M2 B,
      const char side, const char uplo, const char transA);
};

/**
 * Multiple #gdmm.
 */
template<class V1, class M1, class M2>
void multi_gdmm(const int P, const typename V1::value_type alpha, const V1 A, const M1 X,
    const typename M2::value_type beta, M2 Y, const char side = 'L');

/**
 * Multiple #ger.
 *
 * @ingroup math_multi_op
 */
template<class V1, class V2, class M1>
void multi_ger(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
    const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_ger_impl {
  template<class V1, class V2, class M1>
  static void func(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
      const bool clear);
};

/**
 * Multiple #syr.
 *
 * @ingroup math_multi_op
 */
template<class V1, class M1>
void multi_syr(const int P, const typename V1::value_type alpha, const V1 x, M1 A,
    const char uplo = 'U', const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_syr_impl {
  template<class V1, class M1>
  static void func(const int P, const typename V1::value_type alpha, const V1 x, M1 A,
      const char uplo, const bool clear);
};

/**
 * Multiple #syr2
 *
 * @ingroup math_multi_op
 */
template<class V1, class V2, class M1>
void multi_syr2(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
    const char uplo = 'U', const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_syr2_impl {
  template<class V1, class V2, class M1>
  static void func(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
      const char uplo, const bool clear);
};

/**
 * Multiple #syrk.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_syrk(const int P, const typename M1::value_type alpha, const M1 A,
    const typename M2::value_type beta, M2 C, const char uplo = 'U',
    const char trans = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_syrk_impl {
  template<class M1, class M2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A,
      const typename M2::value_type beta, M2 C, const char uplo,
      const char trans);
};

/**
 * Multiple #trsv.
 *
 * @ingroup math_multi_op
 */
template<class M1, class V1>
void multi_trsv(const int P, const M1 A, V1 x, const char uplo = 'U',
    const char trans = 'N', const char diag = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_trsv_impl {
  template<class M1, class V1>
  static void func(const int P, const M1 A, V1 x, const char uplo, const char trans,
      const char diag);
};

/**
 * Multiple #trsm.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_trsm(const int P, const typename M1::value_type alpha, const M1 A, M2 B,
    const char side = 'L', const char uplo = 'U', const char trans = 'N',
    const char diag = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct multi_trsm_impl {
  template<class M1, class M2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, M2 B,
      const char side, const char uplo, const char trans, const char diag);
};
//@}

/**
 * @name LAPACK adapters
 */
//@{
/**
 * Multiple #potrf.
 *
 * @ingroup math_multi_op
 */
template<class M1>
void multi_potrf(const int P, M1 U, char uplo = 'U') throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_potrf_impl {
  template<class M1>
  static void func(const int P, M1 U, char uplo) throw (CholeskyException);
};

/**
 * Multiple #potrs.
 *
 * @ingroup math_multi_op
 */
template<class M1, class M2>
void multi_potrs(const int P, const M1 U, M2 X, char uplo = 'U') throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct multi_potrs_impl {
  template<class M1, class M2>
  static void func(const int P, const M1 U, M2 X, char uplo) throw (CholeskyException);
};
//@}
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/repeated_range.hpp"
#include "../primitive/stuttered_range.hpp"

#include "../host/math/multi_operation.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/math/multi_operation.hpp"
#endif

template<class M1, class M2>
void bi::multi_get_matrix(const int P, const M1 Xs, const int p, M2 X) {
  /* pre-condition */
  BI_ASSERT(P*X.size1() == Xs.size1() && X.size2() == Xs.size2());

  for (int j = 0; j < X.size2(); ++j) {
    column(X, j) = subrange(column(Xs, j), p, X.size1(), P);
  }
}

template<class M1, class M2>
void bi::multi_set_matrix(const int P, M1 Xs, const int p, const M2 X) {
  /* pre-condition */
  BI_ASSERT(P*X.size1() == Xs.size1() && X.size2() == Xs.size2());

  for (int j = 0; j < X.size2(); ++j) {
    subrange(column(Xs, j), p, X.size1(), P) = column(X, j);
  }
}

template<class V1, class V2>
void bi::multi_get_vector(const int P, const V1 xs, const int p, V2 x) {
  /* pre-condition */
  BI_ASSERT(P*x.size() == xs.size());

  x = subrange(xs, p, x.size(), P);
}

template<class V1, class V2>
void bi::multi_set_vector(const int P, V1 xs, const int p, const V2 x) {
  /* pre-condition */
  BI_ASSERT(P*x.size() == xs.size());

  subrange(xs, p, x.size(), P) = x;
}

template<class M1, class M2>
void bi::multi_transpose(const int P, const M1 A, M2 B) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A1(A.size1()/P, A.size2());
    typename sim_temp_matrix<M2>::type B1(B.size1()/P, B.size2());
    int p;

    #pragma omp for
    for (int p = 0; p < P; ++p) {
      multi_get_matrix(P, A, p, A1);
      multi_get_matrix(P, B, p, B1);

      transpose(A1, B1);

      multi_set_matrix(P, B, p, B1);
    }
  }
}

template<class M1, class M2>
void bi::multi_chol(const int P, const M1 A, M2 U, char uplo,
    const CholeskyStrategy strat) throw (CholeskyException) {
  BI_ASSERT(A.size1() == U.size1() && A.size2() == U.size2());

  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A1(A.size1()/P, A.size2());
    typename sim_temp_matrix<M2>::type U1(U.size1()/P, U.size2());
    int p;

    #pragma omp for
    for (int p = 0; p < P; ++p) {
      multi_get_matrix(P, A, p, A1);
      multi_get_matrix(P, U, p, U1);

      chol(A1, U1, uplo, strat);

      multi_set_matrix(P, U, p, U1);
    }
  }
}

template<class M1, class M2>
void bi::multi_matrix_axpy(const int P, const typename M1::value_type a, const M1 X, M2 Y,
    const bool clear) {
  /* pre-conditions */
  BI_ASSERT(X.size1() == Y.size1() && P*X.size2() == P*Y.size2());

  if (X.size1() == X.lead() && Y.size1() == Y.lead()) {
    /* do as one vector axpy */
    multi_axpy(P, a, vec(X), vec(Y), clear);
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < P*X.size2(); ++j) {
      multi_axpy(P, a, column(X,j), column(Y,j), clear);
    }
  }
}

template<class M1>
inline void bi::multi_matrix_scal(const int P, const typename M1::value_type alpha, M1 X) {
  if (X.size1() == X.lead()) {
    /* do as one vector scal */
    scal(alpha, vec(X));
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < P*X.size2(); ++j) {
      multi_scal(P, alpha, column(X,j));
    }
  }
}

template<class V1, class M1, class V2, class M2, class M3, class V3>
void bi::multi_condition(const int P, V1 mu1, M1 U1, const V2 mu2, const M2 U2,
    const M3 C, const V3 x2) {
  /* pre-condition */
  BI_ASSERT(U1.size1() == P*U1.size2());
  BI_ASSERT(U2.size1() == P*U2.size2());
  BI_ASSERT(mu1.size() == U1.size1());
  BI_ASSERT(mu2.size() == U2.size1());
  BI_ASSERT(C.size1() == mu1.size() && P*C.size2() == mu2.size());

  typename sim_temp_vector<V1>::type z2(mu2.size()), b(mu1.size());
  typename sim_temp_matrix<M1>::type K(C.size1(), P*C.size2());

  /**
   * Compute gain matrix:
   *
   * \f[\mathbf{K} = \mathbf{U}_{1}\mathbf{U}_{12}\mathbf{U}_2^{-T}.\f]
   */
  K = C;
  multi_trsm(P, 1.0, U2, K, 'R', 'U');

  /**
   * Update mean:
   *
   * \f[\boldsymbol{\mu}_1 \gets \boldsymbol{\mu}_1 + \mathbf{K}\mathbf{U}_2^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2).\f]
   */
  set_rows(reshape(vector_as_column_matrix(z2), P, z2.size()/P), x2);
  axpy(-1.0, mu2, z2);
  multi_trsv(P, U2, z2, 'U', 'T');
  multi_gemv(P, 1.0, K, z2, 1.0, mu1);

  /**
   * Update Cholesky factor of covariance using downdate, noting:
   *
   * \f[\mathbf{U}_1\mathbf{U}_1^T = \mathbf{U}_{xx}\mathbf{U}_{xx}^T -
   * \mathbf{K}\mathbf{K}^T.\f]
   */
  multi_chkdn(P, U1, K, b);
}

template<class M1, class M2, class V2>
void bi::multi_chkup(const int P, M1 U, M2 A, V2 b) {
  int j;
  for (j = 0; j < A.size2(); ++j) {
    multi_ch1up(P, U, column(A,j), b);
  }
}

template<class M1, class V1, class V2>
void bi::multi_ch1up(const int P, M1 U, V1 a, V2 b) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-condition */
  BI_ASSERT(U.size1() == P*U.size2());
  BI_ASSERT(U.size1() == a.size());
  BI_ASSERT(U.size1() == b.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  multi_ch1up_impl<L,T1>::func(P, U, a, b);
}

template<class M1, class M2, class V2>
void bi::multi_chkdn(const int P, M1 U, M2 A, V2 b) throw (CholeskyException) {
  int j;
  for (j = 0; j < A.size2(); ++j) {
    multi_ch1dn(P, U, column(A,j), b);
  }
}

template<class M1, class V1, class V2>
void bi::multi_ch1dn(const int P, M1 U, V1 a, V2 b) throw (CholeskyException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-condition */
  BI_ASSERT(U.size1() == P*U.size2());
  BI_ASSERT(U.size1() == a.size());
  BI_ASSERT(U.size1() == b.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  multi_ch1dn_impl<L,T1>::func(P, U, a, b);
}

template<class V1>
inline void bi::multi_scal(const int P, typename V1::value_type alpha, V1 x) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;

  multi_scal_impl<L,T1>::func(P, alpha, x);
}

template<class V1, class V2>
inline void bi::multi_axpy(const int P, const typename V1::value_type a, const V1 x, V2 y, const bool clear) {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  BI_ASSERT(V1::on_device == V2::on_device);

  /* pre-conditions */
  BI_ASSERT(x.size() == y.size());
  BI_ASSERT((equals<T1,T2>::value));

  return multi_axpy_impl<L,T2>::func(P, a, x, y, clear);
}

template<class M1, class V1, class V2>
void bi::multi_gemv(const int P, const typename M1::value_type alpha, const M1 A, const V1 x, const typename V2::value_type beta, V2 y, const char transA) {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;
  BI_ASSERT(M1::on_device == V1::on_device);
  BI_ASSERT(V1::on_device == V2::on_device);

  /* pre-conditions */
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(transA != 'N' || (P*A.size2() == x.size() && A.size1() == y.size()));
  BI_ASSERT(transA != 'T' || (A.size1() == x.size() && P*A.size2() == y.size()));
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  multi_gemv_impl<L,T3>::func(P, alpha, A, x, beta, y, transA);
}

template<class M1, class V1, class V2>
void bi::multi_symv(const int P, const typename M1::value_type alpha, const M1 A, const V1 x, const typename V2::value_type beta, V2 y, const char uplo = 'N') {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(P*A.size2() == x.size() && A.size1() == y.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == V1::on_device);
  BI_ASSERT(V1::on_device == V2::on_device);

  multi_symv_impl<L,T3>::func(P, alpha, A, x, beta, y, uplo);
}

template<class M1, class V1>
void bi::multi_trmv(const int P, const M1 A, V1 x, const char uplo, const char transA) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(transA != 'N' || P*A.size2() == x.size());
  BI_ASSERT(transA != 'T' || A.size1() == x.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == V1::on_device);

  multi_trmv_impl<L,T2>::func(P, A, x, uplo, transA);
}

template<class V1, class V2, class V3>
void bi::multi_gdmv(const int P, const typename V1::value_type alpha, const V1 A, const V2 x, const typename V3::value_type beta, V3 y) {
  static const Location L = V3::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename V3::value_type T3;

  /* pre-conditions */
  BI_ASSERT(A.size() == x.size());
  BI_ASSERT(x.size() == y.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(V1::on_device == V2::on_device);
  BI_ASSERT(V2::on_device == V3::on_device);

  multi_gdmv_impl<L,T3>::func(P, alpha, A, x, beta, y);
}

template<class M1, class M2, class M3>
void bi::multi_gemm(const int P, const typename M1::value_type alpha, const M1 A, const M2 X, const typename M3::value_type beta, M3 Y, const char transA, const char transX) {
  static const Location L = M3::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  typedef typename M3::value_type T3;

  /* pre-conditions */
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(transX == 'N' || transX == 'T');
  BI_ASSERT(!(transA == 'N' && transX == 'N') ||
      (P*A.size2() == X.size1() && A.size1() == Y.size1() && P*X.size2() == P*Y.size2()));
  BI_ASSERT(!(transA == 'T' && transX == 'T') ||
      (A.size1() == P*X.size2() && P*A.size2() == Y.size1() && X.size1() == P*Y.size2()));
  BI_ASSERT(!(transA == 'N' && transX == 'T') ||
      (P*A.size2() == P*X.size2() && A.size1() == Y.size1() && X.size1() == P*Y.size2()));
  BI_ASSERT(!(transA == 'T' && transX == 'N') ||
      (A.size1() == X.size1() && P*A.size2() == Y.size1() && P*X.size2() == P*Y.size2()));
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == M2::on_device);
  BI_ASSERT(M2::on_device == M3::on_device);

  multi_gemm_impl<L,T3>::func(P, alpha, A, X, beta, Y, transA, transX);
}

template<class M1, class M2, class M3>
void bi::multi_symm(const int P, const typename M1::value_type alpha, const M1 A, const M2 X, const typename M3::value_type beta, M3 Y, const char side, const char uplo) {
  static const Location L = M3::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  typedef typename M3::value_type T3;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(!(side == 'L') ||
      (P*A.size2() == X.size1() && A.size1() == Y.size1() && P*X.size2() == P*Y.size2()));
  BI_ASSERT(!(side == 'R') ||
      (P*X.size2() == A.size1() && X.size1() == Y.size1() && P*A.size2() == P*Y.size2()));
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == M2::on_device);
  BI_ASSERT(M2::on_device == M3::on_device);

  multi_symm_impl<L,T3>::func(P, alpha, A, X, beta, Y, side, uplo);
}

template<class M1, class M2>
void bi::multi_trmm(const int P, const typename M1::value_type alpha, const M1 A, M2 B, const char side, const char uplo, const char transA) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(transA == 'T' || transA == 'N');
  BI_ASSERT(!(transA == 'N' && side == 'L') || P*A.size2() == B.size1());
  BI_ASSERT(!(transA == 'T' && side == 'L') || A.size1() == B.size1());
  BI_ASSERT(!(transA == 'N' && side == 'R') || P*B.size2() == A.size1());
  BI_ASSERT(!(transA == 'T' && side == 'R') || P*B.size2() == P*A.size2());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  multi_trmm_impl<L,T2>::func(P, alpha, A, B, side, uplo, transA);
}

template<class V1, class M1, class M2>
void bi::multi_gdmm(const int P, const typename V1::value_type alpha, const V1 A, const M1 X,
    const typename M2::value_type beta, M2 Y, const char side) {
  gdmm(alpha, A, X, beta, Y, side);
}

template<class V1, class V2, class M1>
void bi::multi_ger(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename M1::value_type T3;

  /* pre-conditions */
  BI_ASSERT(x.size() == A.size1());
  BI_ASSERT(y.size() == P*A.size2());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(V1::on_device == V2::on_device);
  BI_ASSERT(V2::on_device == M1::on_device);

  multi_ger_impl<L,T3>::func(P, alpha, x, y, A, clear);
}

template<class V1, class M1>
void bi::multi_syr(const int P, const typename V1::value_type alpha, const V1 x, M1 A, const char uplo, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename M1::value_type T2;

  /* pre-condition */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(A.size1() == P*A.size2());
  BI_ASSERT(x.size() == A.size1());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(V1::on_device == M1::on_device);

  multi_syr_impl<L,T2>::func(P, alpha, x, A, uplo, clear);
}

template<class V1, class V2, class M1>
void bi::multi_syr2(const int P, const typename V1::value_type alpha, const V1 x, const V2 y, M1 A, const char uplo, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename M1::value_type T3;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(V1::on_device == V2::on_device);
  BI_ASSERT(V2::on_device == M1::on_device);

  multi_syr2_impl<L,T3>::func(P, alpha, x, y, A, uplo, clear);
}

template<class M1, class M2>
void bi::multi_syrk(const int P, const typename M1::value_type alpha, const M1 A, const typename M2::value_type beta, M2 C, const char uplo, const char trans) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  BI_ASSERT(M1::on_device == M2::on_device);

  /* pre-conditions */
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(C.size1() == P*C.size2());
  BI_ASSERT(trans != 'N' || A.size1() == C.size1());
  BI_ASSERT(trans != 'T' || P*A.size2() == C.size1());
  BI_ASSERT((equals<T1,T2>::value));

  multi_syrk_impl<L,T2>::func(P, alpha, A, beta, C, uplo, trans);
}

template<class M1>
void bi::multi_potrf(const int P, const M1 U, char uplo) throw (CholeskyException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(U.size1() == P*U.size2());

  multi_potrf_impl<L,T1>::func(P, U, uplo);
}

template<class M1, class M2>
void bi::multi_potrs(const int P, const M1 U, M2 X, char uplo) throw (CholeskyException) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(P*U.size2() == X.size1());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  multi_potrs_impl<L,T2>::func(P, U, X, uplo);
}

template<class M1, class V1>
void bi::multi_trsv(const int P, const M1 A, V1 x, const char uplo, const char trans, const char diag) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(diag == 'U' || diag == 'N');
  BI_ASSERT(!(trans == 'T') || A.size1() == x.size());
  BI_ASSERT(!(trans == 'N') || P*A.size2() == x.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == V1::on_device);

  multi_trsv_impl<L,T2>::func(P, A, x, uplo, trans, diag);
}

template<class M1, class M2>
void bi::multi_trsm(const int P, const typename M1::value_type alpha, const M1 A, M2 B, const char side, const char uplo, const char trans, const char diag) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(diag == 'U' || diag == 'N');
  BI_ASSERT(!(trans == 'T' && side == 'L')  || A.size1() == B.size1());
  BI_ASSERT(!(trans == 'N' && side == 'L')  || P*A.size2() == B.size1());
  BI_ASSERT(!(trans == 'T' && side == 'R')  || P*B.size2() == P*A.size2());
  BI_ASSERT(!(trans == 'N' && side == 'R')  || P*B.size2() == A.size1());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  multi_trsm_impl<L,T2>::func(P, alpha, A, B, side, uplo, trans, diag);
}

#endif
