/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_OPERATION_HPP
#define BI_MATH_OPERATION_HPP

#include "../misc/exception.hpp"
#include "../misc/location.hpp"

namespace bi {
/**
 * Strategies for handling singular matrices in chol().
 *
 * @ingroup math-op
 */
enum CholeskyStrategy {
  /**
   * Adjust diagonal with small increments.
   */
  ADJUST_DIAGONAL,

  /**
   * Use eigendecomposition and eliminate negatives.
   */
  ZERO_NEGATIVE_EIGENVALUES,

  /**
   * Do nothing, fail.
   */
  FAIL
};

/**
 * Write identity matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 *
 * @param[out] A Matrix.
 *
 * Zeros the given matrix except for ones along the leading diagonal.
 */
template<class M1>
void ident(M1 A);

/**
 * Transpose matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param A Matrix.
 * @param[out] B Matrix.
 *
 * Writes the transpose of @p A into @p B.
 */
template<class M1, class M2>
void transpose(const M1 A, M2 B);

/**
 * Symmetric matrix Cholesky decomposition.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void chol(const M1 A, M2 U, char uplo = 'U', const CholeskyStrategy strat =
    ADJUST_DIAGONAL) throw (CholeskyException);

/**
 * Scalar multiply and matrix add.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void matrix_axpy(const typename M1::value_type a, const M1 X, M2 Y,
    const bool clear = false);

/**
 * Matrix scale.
 *
 * @ingroup math_op
 */
template<class M1>
void matrix_scal(typename M1::value_type alpha, M1 X);

/**
 * Condition Gaussian distribution.
 *
 * @ingroup math_op
 *
 * Considers a Gaussian distribution over a partitioned set
 * of variables \f$\{X_1,X_2\}\f$, \f$|X_1| = M\f$, \f$|X_2| = N\f$. For a
 * sample \f$\mathbf{x}_2\f$, computes \f$p(X_1|\mathbf{x}_2)\f$.
 *
 * @param[in,out] mu1 \f$\boldsymbol{\mu}_1\f$; mean of first partition.
 * @param[in,out] U1 \f$\mathbf{U}_1\f$; Cholesky factor of covariance matrix
 * of first partition.
 * @param mu2 \f$\boldsymbol{\mu}_2\f$; mean of second partition.
 * @param U2 \f$\mathbf{U}_2\f$; Cholesky factor of covariance matrix of
 * second partition.
 * @param C \f$\mathbf{C} = \mathbf{U}_1\mathbf{U}_{12}\f$; Cross-covariance
 * matrix between \f$X_1\f$ and \f$X_2\f$, %size \f$M \times N\f$.
 * @param x2 \f$\mathbf{x}_2\f$.
 */
template<class V1, class M1, class V2, class M2, class M3, class V3>
void condition(V1 mu1, M1 U1, const V2 mu2, const M2 U2, const M3 C,
    const V3 x2);

/**
 * Marginalise Gaussian distribution.
 *
 * @ingroup math_pdf
 *
 * Considers a Gaussian distribution over a partitioned set
 * of variables \f$\{X_1,X_2\}\f$, \f$|X_1| = M\f$, \f$|X_2| = N\f$. For a
 * distribution \f$q(\mathbf{X}_2)\f$, computes
 * \f$\int_{-\infty}^{\infty} p(X_1|\mathbf{x}_2)
 * q(\mathbf{x}_2) \,d\mathbf{x}_2\f$.
 *
 * @param p1 \f$p(X_1)\f$; marginal of variables in first partition.
 * @param p2 \f$p(X_2)\f$; marginal of variables in second
 * partition.
 * @param C \f$C_{\mathbf{x}_1,\mathbf{x}_2}\f$; cross-covariance matrix
 * between \f$X_1\f$ and \f$X_2\f$, %size \f$M \times N\f$.
 * @param q2 \f$q(X_2)\f$.
 * @param[out] p3 \f$\int_{-\infty}^{\infty} p(X_2|\mathbf{x}_1)
 * p(\mathbf{x}_1) \,d\mathbf{x}_1\f$.
 */
template<class V1, class M1, class V2, class M2, class M3, class V4, class M4>
void marginalise(V1 mu1, M1 U1, const V2 mu2, const M2 U2, const M3 C,
    const V4 mu3, const M4 U3);

/**
 * @name QRUpdate (and QRUpdate-like) adapters
 */
//@{
/**
 * Rank-k update of upper triangular Cholesky factor.
 *
 * @ingroup math_op
 *
 * @see dch1up, sch1up of qrupdate.
 */
template<class M1, class M2, class V2>
void chkup(M1 U, M2 A, V2 b);

/**
 * Rank-1 update of upper triangular Cholesky factor.
 *
 * @ingroup math_op
 *
 * @see dch1up, sch1up of qrupdate.
 */
template<class M1, class V1, class V2>
void ch1up(M1 U, V1 a, V2 b);

/**
 * @internal
 */
template<Location L, class T1>
struct ch1up_impl {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b);
};

/**
 * Rank-k downdate of upper triangular Cholesky factor.
 *
 * @ingroup math_op
 *
 * @see dch1dn, sch1dn of qrupdate.
 */
template<class M1, class M2, class V2>
void chkdn(M1 U, M2 A, V2 b) throw (CholeskyException);

/**
 * Rank-1 downdate of upper triangular Cholesky factor.
 *
 * @ingroup math_op
 *
 * @see dch1dn, sch1dn of qrupdate.
 */
template<class M1, class V1, class V2>
void ch1dn(M1 U, V1 a, V2 b) throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct ch1dn_impl {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b) throw (CholeskyException);
};

//@}
/**
 * @name BLAS (and BLAS-like) adapters
 */
//@{
/**
 * Vector scale.
 *
 * @ingroup math_op
 */
template<class V1>
void scal(typename V1::value_type alpha, V1 x);

/**
 * @internal
 */
template<Location L, class T1>
struct scal_impl {
  template<class V1>
  static void func(T1 alpha, V1 x);
};

/**
 * Vector dot product.
 *
 * @ingroup math_op
 */
template<class V1, class V2>
typename V1::value_type dot(const V1 a, const V2 b);

/**
 * Vector dot product, with self.
 *
 * @ingroup math_op
 */
template<class V1>
typename V1::value_type dot(const V1 a);

/**
 * @internal
 */
template<Location L, class T1>
struct dot_impl {
  template<class V1, class V2>
  static T1 func(const V1 a, const V2 b);

  template<class V1>
  static T1 func(const V1 a);
};

/**
 * Index of element of vector with largest absolute value.
 *
 * @ingroup math_op
 *
 * @param x Vector.
 *
 * @return Index of element of vector with largest absolute value, zero
 * based.
 */
template<class V1>
typename V1::size_type iamax(const V1 x);

/**
 * @internal
 */
template<Location L, class T1>
struct iamax_impl {
  template<class V1>
  static typename V1::size_type func(const V1 x);
};

/**
 * Scalar multiply and vector add.
 *
 * @ingroup math_op
 */
template<class V1, class V2>
void axpy(const typename V1::value_type a, const V1 x, V2 y,
    const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct axpy_impl {
  template<class V1, class V2>
  static void func(const T1 a, const V1 x, V2 y, const bool clear = false);
};

/**
 * Matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class M1, class V1, class V2>
void gemv(const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct gemv_impl {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char transA);
};

/**
 * Symmetric matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class M1, class V1, class V2>
void symv(const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char uplo = 'U');

/**
 * @internal
 */
template<Location L, class T1>
struct symv_impl {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char uplo);
};

/**
 * Triangular matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void trmv(const M1 A, V1 x, const char uplo = 'U', const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct trmv_impl {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char transA);
};

/**
 * Diagonal matrix-vector multiply.
 *
 * @ingroup math_op
 *
 * Uses @c gbmv internally, with single leading diagonal band.
 */
template<class V1, class V2, class V3>
void gdmv(const typename V1::value_type alpha, const V1 A, const V2 x,
    const typename V3::value_type beta, V3 y);

/**
 * @internal
 */
template<Location L, class T1>
struct gdmv_impl {
  template<class V1, class V2, class V3>
  static void func(const T1 alpha, const V1 A, const V2 x, const T1 beta,
      V3 y);
};

/**
 * Matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class M1, class M2, class M3>
void gemm(const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char transA = 'N',
    const char transX = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct gemm_impl {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char transA, const char transX);
};

/**
 * Symmetric matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class M1, class M2, class M3>
void symm(const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char side = 'L',
    const char uplo = 'U');

/**
 * @internal
 */
template<Location L, class T1>
struct symm_impl {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char side, const char uplo);
};

/**
 * Triangular matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void trmm(const typename M1::value_type alpha, const M1 A, M2 B,
    const char side = 'L', const char uplo = 'U', const char transA = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct trmm_impl {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char transA);
};

/**
 * Diagonal matrix-matrix multiply.
 *
 * @ingroup math_op
 *
 * For diagonal matrix on left side, uses multiple calls to #gdmv on
 * columns of @p X and @p Y internally. For diagonal matrix on right side,
 * uses multiple calls to #scal and #axpy on columns of @p X and @p Y
 * internally.
 */
template<class V1, class M1, class M2>
void gdmm(const typename V1::value_type alpha, const V1 A, const M1 X,
    const typename M2::value_type beta, M2 Y, const char side = 'L');

/**
 * Vector outer product and matrix add.
 *
 * @ingroup math_op
 */
template<class V1, class V2, class M1>
void ger(const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
    const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct ger_impl {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const bool clear);
};

/**
 * Symmetric vector outer product and matrix add.
 *
 * @ingroup math_op
 */
template<class V1, class M1>
void syr(const typename V1::value_type alpha, const V1 x, M1 A,
    const char uplo = 'U', const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct syr_impl {
  template<class V1, class M1>
  static void func(const T1 alpha, const V1 x, M1 A, const char uplo,
      const bool clear);
};

/**
 * Symmetric matrix rank-2 update.
 *
 * @ingroup math_op
 */
template<class V1, class V2, class M1>
void syr2(const typename V1::value_type alpha, const V1 x, const V2 y, M1 A,
    const char uplo = 'U', const bool clear = false);

/**
 * @internal
 */
template<Location L, class T1>
struct syr2_impl {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const char uplo, const bool clear);
};

/**
 * Matrix rank-k update.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void syrk(const typename M1::value_type alpha, const M1 A,
    const typename M2::value_type beta, M2 C, const char uplo = 'U',
    const char trans = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct syrk_impl {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, const T1 beta, M2 C,
      const char uplo, const char trans);
};

/**
 * Triangular linear system solve.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void trsv(const M1 A, V1 x, const char uplo = 'U', const char trans = 'N',
    const char diag = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct trsv_impl {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char trans,
      const char diag);
};

/**
 * Triangular linear system solve.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void trsm(const typename M1::value_type alpha, const M1 A, M2 B,
    const char side = 'L', const char uplo = 'U', const char trans = 'N',
    const char diag = 'N');

/**
 * @internal
 */
template<Location L, class T1>
struct trsm_impl {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char trans, const char diag);
};
//@}

/**
 * @name LAPACK adapters
 */
//@{
/**
 * Cholesky factorisation.
 *
 * @ingroup math_op
 *
 * @seealso chol
 */
template<class M1>
void potrf(M1 U, char uplo = 'U') throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct potrf_impl {
  template<class M1>
  static void func(M1 U, char uplo) throw (CholeskyException);
};

/**
 * Symmetric positive definite linear system solve.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void potrs(const M1 U, M2 X, char uplo = 'U') throw (CholeskyException);

/**
 * @internal
 */
template<Location L, class T1>
struct potrs_impl {
  template<class M1, class M2>
  static void func(const M1 U, M2 X, char uplo) throw (CholeskyException);
};

/**
 * Eigenvalues and eigenvectors.
 *
 * @ingroup math_op
 */
template<class M1, class V1, class M2, class V2, class V3, class V4, class V5>
void syevx(char jobz, char range, char uplo, M1 A, typename M1::value_type vl,
    typename M1::value_type vu, int il, int iu,
    typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work, V3 rwork,
    V4 iwork, V5 ifail) throw (EigenException);

/**
 * @internal
 */
template<Location L, class T1>
struct syevx_impl {
  template<class M1, class V1, class M2, class V2, class V3, class V4,
      class V5>
  static void func(const char jobz, const char range, const char uplo, M1 A,
      const typename M1::value_type vl, const typename M1::value_type vu,
      const int il, const int iu, const typename M1::value_type abstol,
      int* m, V1 w, M2 Z, V2 work, V3 rwork, V4 iwork, V5 ifail)
          throw (EigenException);
};
//@}
}

#include "view.hpp"
#include "sim_temp_vector.hpp"
#include "sim_temp_matrix.hpp"
#include "loc_temp_vector.hpp"
#include "loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"
#include "../typelist/equals.hpp"
#include "../host/math/operation.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/math/operation.hpp"
#endif

template<class M1>
inline void bi::ident(M1 A) {
  BOOST_AUTO(d, diagonal(A));
  A.clear();
  set_elements(d, static_cast<typename M1::value_type>(1.0));
}

template<class M1, class M2>
inline void bi::transpose(const M1 A, M2 B) {
  /* pre-condition */
  BI_ASSERT(A.size1() == B.size2() && A.size2() == B.size1());

  int i;
  for (i = 0; i < A.size1(); ++i) {
    column(B.ref(), i) = row(A.ref(), i);
  }
}

template<class M1, class M2>
void bi::chol(const M1 A, M2 U, char uplo, const CholeskyStrategy strat)
    throw (CholeskyException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  typedef typename loc_temp_vector<L,T1>::type temp_vector_type;
  typedef typename loc_temp_vector<L,int>::type temp_int_vector_type;
  typedef typename loc_temp_matrix<L,T1>::type temp_matrix_type;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(A.size1() == U.size1());
  BI_ASSERT(A.size2() == U.size2());
  BI_ASSERT(U.size1() == U.size2());
  BI_ASSERT(U.inc() == 1);

  const int N = A.size1();

  if (uplo == 'U') {
    set_upper_triangle(U, A);
  } else {
    set_lower_triangle(U, A);
  }
  try {
    potrf(U, uplo);
  } catch (CholeskyException e) {
    BOOST_AUTO(d, diagonal(U));
    T1 largest = amax_reduce(d);

    if (!is_finite(largest)) {
      throw e;
    } else if (largest <= 0.0) {
      U.clear();
    } else if (strat == ADJUST_DIAGONAL) {
      bool success = false;
      T1 smallest = amin_reduce(d);
      T1 factor;
      if (1.0e-9 * smallest > 0.0) {
        factor = 1.0e-9 * smallest;
      } else {
        factor = 1.0e-9;
      }

      while (!success) {
        if (uplo == 'U') {
          set_upper_triangle(U, A);
        } else {
          set_lower_triangle(U, A);
        }
        addscal_elements(d, factor, d);
        try {
          potrf(U, uplo);
          success = true;
        } catch (CholeskyException e) {
          factor *= 2.0;
          if (factor > largest) {
            throw e;
          }
        }
      }
    } else if (strat == ZERO_NEGATIVE_EIGENVALUES) {
      temp_vector_type w(N), work(8 * N), rwork(8 * N);  ///@todo Query for optimal size of work, see LAPACK docs
      temp_int_vector_type iwork(5 * N), ifail(N);
      temp_matrix_type Z(N, N), Sigma(N, N);
      int m;

      syevx('V', 'A', uplo, A, 0.0, 0.0, 0, 0, 0.0, &m, w, Z, work, rwork,
          iwork, ifail);
      maxscal_elements(w, 0.0, w);
      sqrt_elements(w, w);
      gdmm(1.0, w, Z, 0.0, U);
      syrk(1.0, U, 0.0, Sigma, 'U', 'T');
      chol(Sigma, U, 'U', FAIL);
    } else {
      throw e;
    }
  }
}

template<class M1, class M2>
void bi::matrix_axpy(const typename M1::value_type a, const M1 X, M2 Y,
    const bool clear) {
  /* pre-conditions */
  BI_ASSERT(X.size1() == Y.size1() && X.size2() == Y.size2());

  if (X.can_vec() && Y.can_vec()) {
    /* do as one vector axpy */
    axpy(a, vec(X), vec(Y), clear);
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < X.size2(); ++j) {
      axpy(a, column(X, j), column(Y, j), clear);
    }
  }
}

template<class M1>
inline void bi::matrix_scal(const typename M1::value_type alpha, M1 X) {
  if (X.can_vec()) {
    /* do as one vector scal */
    scal(alpha, vec(X));
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < X.size2(); ++j) {
      scal(alpha, column(X, j));
    }
  }
}

template<class V1, class M1, class V2, class M2, class M3, class V3>
void bi::condition(V1 mu1, M1 U1, const V2 mu2, const M2 U2, const M3 C,
    const V3 x2) {
  /* pre-condition */
  BI_ASSERT(U1.size1() == U1.size2());
  BI_ASSERT(U2.size1() == U2.size2());
  BI_ASSERT(mu1.size() == U1.size1());
  BI_ASSERT(mu2.size() == U2.size1());
  BI_ASSERT(C.size1() == mu1.size() && C.size2() == mu2.size());

  typename sim_temp_vector<V1>::type z2(mu2.size()), b(mu1.size());
  typename sim_temp_matrix<M1>::type K(mu1.size(), mu2.size());

  /**
   * Compute gain matrix:
   *
   * \f[\mathbf{K} = \mathbf{C}\mathbf{U}_2^{-1}.\f]
   */
  K = C;
  trsm(1.0, U2, K, 'R', 'U');

  /**
   * Update mean:
   *
   * \f[\boldsymbol{\mu}_1 \gets \boldsymbol{\mu}_1 + \mathbf{K}\mathbf{U}_2^{-T}(\mathbf{x}_2 - \boldsymbol{\mu}_2).\f]
   */
  sub_elements(x2, mu2, z2);
  trsv(U2, z2, 'U', 'T');
  gemv(1.0, K, z2, 1.0, mu1);

  /**
   * Update Cholesky factor of covariance using downdate, noting:
   *
   * \f[\mathbf{U}_1^T\mathbf{U}_1 \gets \mathbf{U}_{1}^T\mathbf{U}_{1} -
   * \mathbf{K}\mathbf{K}^T.\f]
   */
  if (K.size2() < U1.size2()) {
    try {
      chkdn(U1, K, b);
    } catch (CholeskyException e) {
      typename sim_temp_matrix<M1>::type Sigma1(U1.size1(), U1.size2());
      Sigma1.clear();
      syrk(1.0, U1, 0.0, Sigma1, 'U', 'T');
      syrk(-1.0, K, 1.0, Sigma1, 'U', 'N');

      chol(Sigma1, U1, 'U');
    }
  } else {
    typename sim_temp_matrix<M1>::type Sigma1(U1.size1(), U1.size2());
    Sigma1.clear();
    syrk(1.0, U1, 0.0, Sigma1, 'U', 'T');
    syrk(-1.0, K, 1.0, Sigma1, 'U', 'N');

    chol(Sigma1, U1, 'U');
  }
}

template<class V1, class M1, class V2, class M2, class M3, class V4, class M4>
void bi::marginalise(V1 mu1, M1 U1, const V2 mu2, const M2 U2, const M3 C,
    const V4 mu3, const M4 U3) {
  /* pre-conditions */
  BI_ASSERT(U1.size1() == U1.size2());
  BI_ASSERT(U2.size1() == U2.size2());
  BI_ASSERT(U3.size1() == U3.size2());
  BI_ASSERT(mu1.size() == U1.size1());
  BI_ASSERT(mu2.size() == U2.size1());
  BI_ASSERT(mu3.size() == U3.size1());
  BI_ASSERT(C.size1() == mu1.size1() && C.size2() == mu2.size1());
  BI_ASSERT(mu3.size() == mu2.size());

  typename sim_temp_vector<V1>::type z2(mu2.size()), b(mu1.size());
  typename sim_temp_matrix<M1>::type K(mu1.size(), mu2.size());

  /**
   * Compute gain matrix:
   *
   * \f[\mathbf{K} = \mathbf{U}_{1}\mathbf{U}_{12}\mathbf{U}_2^{-T}.\f]
   */
  K = C;
  trsm(1.0, U2, K, 'R', 'U', 'T');

  /**
   * Update mean:
   *
   * \f[\boldsymbol{\mu}_1 \gets \boldsymbol{\mu}_1 + \mathbf{K}\mathbf{U}_2^{-1}(\boldsymbol{\mu}_3 - \boldsymbol{\mu}_2).\f]
   */
  z2 = mu3;
  axpy(-1.0, mu2, z2);
  trsv(U2, mu1, 'U');
  gemv(1.0, K, z2, 1.0, mu1);

  /**
   * Update Cholesky factor of covariance:
   *
   * \f[\mathbf{U}_1\mathbf{U}_1^T = \mathbf{U}_{xx}\mathbf{U}_{xx}^T -
   * \mathbf{K}\mathbf{K}^T +
   * \mathbf{K}\mathbf{U}_3^T\mathbf{U}_3\mathbf{K}^T.\f]
   */
  chkdn(U1, K, b);
  trmm(1.0, U3, K, 'R', 'U', 'T');
  chkup(U1, K, b);
}

template<class M1, class M2, class V2>
void bi::chkup(M1 U, M2 A, V2 b) {
  int j;
  for (j = 0; j < A.size2(); ++j) {
    ch1up(U, column(A, j), b);
  }
}

template<class M1, class V1, class V2>
void bi::ch1up(M1 U, V1 a, V2 b) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-condition */
  BI_ASSERT(U.size1() == U.size2());
  BI_ASSERT(U.size1() == a.size());
  BI_ASSERT(U.size1() == b.size());
  BI_ASSERT(U.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  ch1up_impl<L,T1>::func(U, a, b);
}

template<class M1, class M2, class V2>
void bi::chkdn(M1 U, M2 A, V2 b) throw (CholeskyException) {
  int j;
  for (j = 0; j < A.size2(); ++j) {
    ch1dn(U, column(A, j), b);
  }
}

template<class M1, class V1, class V2>
void bi::ch1dn(M1 U, V1 a, V2 b) throw (CholeskyException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-condition */
  BI_ASSERT(U.size1() == U.size2());
  BI_ASSERT(U.size1() == a.size());
  BI_ASSERT(U.size1() == b.size());
  BI_ASSERT(U.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  ch1dn_impl<L,T1>::func(U, a, b);
}

template<class V1>
inline void bi::scal(typename V1::value_type alpha, V1 x) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;

  scal_impl<L,T1>::func(alpha, x);
}

template<class V1, class V2>
inline typename V1::value_type bi::dot(const V1 a, const V2 b) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(a.size() == b.size());
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T1,float>::value || equals<T1,double>::value));
  BI_ASSERT(V1::on_device == V2::on_device);

  return dot_impl<L,T1>::func(a, b);
}

template<class V1>
inline typename V1::value_type bi::dot(const V1 a) {
  return dot(a, a);
}

template<class V1>
inline typename V1::size_type bi::iamax(const V1 x) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;

  return iamax_impl<L,T1>::func(x);
}

template<class V1, class V2>
inline void bi::axpy(const typename V1::value_type a, const V1 x, V2 y,
    const bool clear) {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  BI_ASSERT(V1::on_device == V2::on_device);

  /* pre-conditions */
  BI_ASSERT(x.size() == y.size());
  BI_ASSERT((equals<T1,T2>::value));

  return axpy_impl<L,T2>::func(a, x, y, clear);
}

template<class M1, class V1, class V2>
void bi::gemv(const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char transA) {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;
  BI_ASSERT(M1::on_device == V1::on_device);
  BI_ASSERT(V1::on_device == V2::on_device);

  /* pre-conditions */
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(
      transA != 'N' || (A.size2() == x.size() && A.size1() == y.size()));
  BI_ASSERT(
      transA != 'T' || (A.size1() == x.size() && A.size2() == y.size()));
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));

  gemv_impl<L,T3>::func(alpha, A, x, beta, y, transA);
}

template<class M1, class V1, class V2>
void bi::symv(const typename M1::value_type alpha, const M1 A, const V1 x,
    const typename V2::value_type beta, V2 y, const char uplo) {
  static const Location L = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(A.size2() == x.size() && A.size1() == y.size());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == V1::on_device);
  BI_ASSERT(V1::on_device == V2::on_device);

  symv_impl<L,T3>::func(alpha, A, x, beta, y, uplo);
}

template<class M1, class V1>
void bi::trmv(const M1 A, V1 x, const char uplo, const char transA) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(transA != 'N' || A.size2() == x.size());
  BI_ASSERT(transA != 'T' || A.size1() == x.size());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == V1::on_device);

  trmv_impl<L,T2>::func(A, x, uplo, transA);
}

template<class V1, class V2, class V3>
void bi::gdmv(const typename V1::value_type alpha, const V1 A, const V2 x,
    const typename V3::value_type beta, V3 y) {
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

  gdmv_impl<L,T3>::func(alpha, A, x, beta, y);
}

template<class M1, class M2, class M3>
void bi::gemm(const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char transA,
    const char transX) {
  static const Location L = M3::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  typedef typename M3::value_type T3;

  /* pre-conditions */
  BI_ASSERT(transA == 'N' || transA == 'T');
  BI_ASSERT(transX == 'N' || transX == 'T');
  BI_ASSERT(
      !(transA == 'N' && transX == 'N')
          || (A.size2() == X.size1() && A.size1() == Y.size1()
              && X.size2() == Y.size2()));
  BI_ASSERT(
      !(transA == 'T' && transX == 'T')
          || (A.size1() == X.size2() && A.size2() == Y.size1()
              && X.size1() == Y.size2()));
  BI_ASSERT(
      !(transA == 'N' && transX == 'T')
          || (A.size2() == X.size2() && A.size1() == Y.size1()
              && X.size1() == Y.size2()));
  BI_ASSERT(
      !(transA == 'T' && transX == 'N')
          || (A.size1() == X.size1() && A.size2() == Y.size1()
              && X.size2() == Y.size2()));
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT(X.inc() == 1);
  BI_ASSERT(Y.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == M2::on_device);
  BI_ASSERT(M2::on_device == M3::on_device);

  gemm_impl<L,T3>::func(alpha, A, X, beta, Y, transA, transX);
}

template<class M1, class M2, class M3>
void bi::symm(const typename M1::value_type alpha, const M1 A, const M2 X,
    const typename M3::value_type beta, M3 Y, const char side,
    const char uplo) {
  static const Location L = M3::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  typedef typename M3::value_type T3;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(
      !(side == 'L')
          || (A.size2() == X.size1() && A.size1() == Y.size1()
              && X.size2() == Y.size2()));
  BI_ASSERT(
      !(side == 'R')
          || (X.size2() == A.size1() && X.size1() == Y.size1()
              && A.size2() == Y.size2()));
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT(X.inc() == 1);
  BI_ASSERT(Y.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(M1::on_device == M2::on_device);
  BI_ASSERT(M2::on_device == M3::on_device);

  symm_impl<L,T3>::func(alpha, A, X, beta, Y, side, uplo);
}

template<class M1, class M2>
void bi::trmm(const typename M1::value_type alpha, const M1 A, M2 B,
    const char side, const char uplo, const char transA) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(transA == 'T' || transA == 'N');
  BI_ASSERT(!(transA == 'N' && side == 'L') || A.size2() == B.size1());
  BI_ASSERT(!(transA == 'T' && side == 'L') || A.size1() == B.size1());
  BI_ASSERT(!(transA == 'N' && side == 'R') || B.size2() == A.size1());
  BI_ASSERT(!(transA == 'T' && side == 'R') || B.size2() == A.size2());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT(B.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  trmm_impl<L,T2>::func(alpha, A, B, side, uplo, transA);
}

template<class V1, class M1, class M2>
void bi::gdmm(const typename V1::value_type alpha, const V1 A, const M1 X,
    const typename M2::value_type beta, M2 Y, const char side) {
  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(
      side != 'L'
          || (A.size() == Y.size1() && Y.size1() == A.size()
              && Y.size2() == X.size2()));
  BI_ASSERT(
      side != 'R'
          || (X.size2() == A.size() && Y.size1() == X.size1()
              && Y.size2() == A.size()));

  if (side == 'L') {
    /* gdmv on each column */
    for (int j = 0; j < X.size2(); ++j) {
      gdmv(alpha, A, column(X, j), beta, column(Y, j));
    }
  } else {
    /* gdmv on each row */
    ///@todo Improve cache use
    for (int i = 0; i < X.size1(); ++i) {
      gdmv(alpha, A, row(X, i), beta, row(Y, i));
    }
  }
}

template<class V1, class V2, class M1>
void bi::ger(const typename V1::value_type alpha, const V1 x, const V2 y,
    M1 A, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename M1::value_type T3;

  /* pre-conditions */
  BI_ASSERT(x.size() == A.size1());
  BI_ASSERT(y.size() == A.size2());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(V1::on_device == V2::on_device);
  BI_ASSERT(V2::on_device == M1::on_device);

  ger_impl<L,T3>::func(alpha, x, y, A, clear);
}

template<class V1, class M1>
void bi::syr(const typename V1::value_type alpha, const V1 x, M1 A,
    const char uplo, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename M1::value_type T2;

  /* pre-condition */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(A.size1() == A.size2());
  BI_ASSERT(x.size() == A.size1());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(V1::on_device == M1::on_device);

  syr_impl<L,T2>::func(alpha, x, A, uplo, clear);
}

template<class V1, class V2, class M1>
void bi::syr2(const typename V1::value_type alpha, const V1 x, const V2 y,
    M1 A, const char uplo, const bool clear) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename M1::value_type T3;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT((equals<T2,T3>::value));
  BI_ASSERT(V1::on_device == V2::on_device);
  BI_ASSERT(V2::on_device == M1::on_device);

  syr2_impl<L,T3>::func(alpha, x, y, A, uplo, clear);
}

template<class M1, class M2>
void bi::syrk(const typename M1::value_type alpha, const M1 A,
    const typename M2::value_type beta, M2 C, const char uplo,
    const char trans) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;
  BI_ASSERT(M1::on_device == M2::on_device);

  /* pre-conditions */
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(C.size1() == C.size2());
  BI_ASSERT(trans != 'N' || A.size1() == C.size1());
  BI_ASSERT(trans != 'T' || A.size2() == C.size1());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT(C.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));

  syrk_impl<L,T2>::func(alpha, A, beta, C, uplo, trans);
}

template<class M1, class V1>
void bi::trsv(const M1 A, V1 x, const char uplo, const char trans, const
char diag) {
  static const Location L = V1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(diag == 'U' || diag == 'N');
  BI_ASSERT(!(trans == 'T') || A.size1() == x.size());
  BI_ASSERT(!(trans == 'N') || A.size2() == x.size());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == V1::on_device);

  trsv_impl<L,T2>::func(A, x, uplo, trans, diag);
}

template<class M1, class M2>
void bi::trsm(const typename M1::value_type alpha, const M1 A, M2 B,
    const char side, const char uplo, const char trans, const char diag) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(side == 'L' || side == 'R');
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(trans == 'N' || trans == 'T');
  BI_ASSERT(diag == 'U' || diag == 'N');
  BI_ASSERT(!(trans == 'T' && side == 'L') || A.size1() == B.size1());
  BI_ASSERT(!(trans == 'N' && side == 'L') || A.size2() == B.size1());
  BI_ASSERT(!(trans == 'T' && side == 'R') || B.size2() == A.size2());
  BI_ASSERT(!(trans == 'N' && side == 'R') || B.size2() == A.size1());
  BI_ASSERT(A.inc() == 1);
  BI_ASSERT(B.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  trsm_impl<L,T2>::func(alpha, A, B, side, uplo, trans, diag);
}

template<class M1>
void bi::potrf(const M1 U, char uplo) throw (CholeskyException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(U.size1() == U.size2());
  BI_ASSERT(U.inc() == 1);

  potrf_impl<L,T1>::func(U, uplo);
}

template<class M1, class M2>
void bi::potrs(const M1 U, M2 X, char uplo) throw (CholeskyException) {
  static const Location L = M2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  BI_ASSERT(uplo == 'U' || uplo == 'L');
  BI_ASSERT(U.size2() == X.size1());
  BI_ASSERT(U.inc() == 1);
  BI_ASSERT(X.inc() == 1);
  BI_ASSERT((equals<T1,T2>::value));
  BI_ASSERT(M1::on_device == M2::on_device);

  potrs_impl<L,T2>::func(U, X, uplo);
}

template<class M1, class V1, class M2, class V2, class V3, class V4, class V5>
void bi::syevx(char jobz, char range, char uplo, M1 A,
    typename M1::value_type vl, typename M1::value_type vu, int il, int iu,
    typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work, V3 rwork,
    V4 iwork, V5 ifail) throw (EigenException) {
  static const Location L = M1::on_device ? ON_DEVICE : ON_HOST;
  typedef typename M1::value_type T1;

  /* pre-conditions */
  BI_ASSERT(jobz == 'N' || jobz == 'V');
  BI_ASSERT(range == 'A' || range == 'V' || range == 'I');
  BI_ASSERT(uplo == 'U' || uplo == 'L');

  syevx_impl<L,T1>::func(jobz, range, uplo, A, vl, vu, il, iu, abstol, m, w,
      Z, work, rwork, iwork, ifail);
}

#endif
