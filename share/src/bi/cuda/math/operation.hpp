/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_OPERATION_HPP
#define BI_CUDA_MATH_OPERATION_HPP

namespace bi {
/**
 * @internal
 */
template<class T1>
struct ch1up_impl<ON_DEVICE,T1> {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b);
};

/**
 * @internal
 */
template<class T1>
struct ch1dn_impl<ON_DEVICE,T1> {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct scal_impl<ON_DEVICE,T1> {
  template<class V1>
  static void func(T1 alpha, V1 x);
};

/**
 * @internal
 */
template<class T1>
struct dot_impl<ON_DEVICE,T1> {
  template<class V1, class V2>
  static T1 func(const V1 a, const V2 b);
};

/**
 * @internal
 */
template<class T1>
struct iamax_impl<ON_DEVICE,T1> {
  template<class V1>
  static typename V1::size_type func(const V1 x);
};

/**
 * @internal
 */
template<class T1>
struct axpy_impl<ON_DEVICE,T1> {
  template<class V1, class V2>
  static void func(const T1 a, const V1 x, V2 y, const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct gemv_impl<ON_DEVICE,T1> {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct symv_impl<ON_DEVICE,T1> {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char uplo);
};

/**
 * @internal
 */
template<class T1>
struct trmv_impl<ON_DEVICE,T1> {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct gdmv_impl<ON_DEVICE,T1> {
  template<class V1, class V2, class V3>
  static void func(const T1 alpha, const V1 A, const V2 x, const T1 beta,
      V3 y);
};

/**
 * @internal
 */
template<class T1>
struct gemm_impl<ON_DEVICE,T1> {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char transA, const char transX);
};

/**
 * @internal
 */
template<class T1>
struct symm_impl<ON_DEVICE,T1> {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char side, const char uplo);
};

/**
 * @internal
 */
template<class T1>
struct trmm_impl<ON_DEVICE,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct ger_impl<ON_DEVICE,T1> {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syr_impl<ON_DEVICE,T1> {
  template<class V1, class M1>
  static void func(const T1 alpha, const V1 x, M1 A, const char uplo,
      const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syr2_impl<ON_DEVICE,T1> {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const char uplo, const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syrk_impl<ON_DEVICE,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, const T1 beta, M2 C,
      const char uplo, const char trans);
};

/**
 * @internal
 */
template<class T1>
struct trsv_impl<ON_DEVICE,T1> {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char trans,
      const char diag);
};

/**
 * @internal
 */
template<class T1>
struct trsm_impl<ON_DEVICE,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char trans, const char diag);
};

/**
 * @internal
 */
template<class T1>
struct potrf_impl<ON_DEVICE,T1> {
  template<class M1>
  static void func(M1 U, char uplo) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct potrs_impl<ON_DEVICE,T1> {
  template<class M1, class M2>
  static void func(const M1 U, M2 X, char uplo) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct syevx_impl<ON_DEVICE,T1> {
  template<class M1, class V1, class M2, class V2, class V3, class V4,
      class V5>
  static void func(char jobz, char range, char uplo, M1 A,
      typename M1::value_type vl, typename M1::value_type vu, int il, int iu,
      typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work, V3 rwork,
      V4 iwork, V5 ifail) throw (EigenException);
};

}

#include "cublas.hpp"

template<class T1>
template<class M1, class V1, class V2>
void bi::ch1up_impl<bi::ON_DEVICE,T1>::func(M1 U, V1 a, V2 b) {
  BI_ERROR_MSG(false, "ch1up not implemented for device");
}

template<class T1>
template<class M1, class V1, class V2>
void bi::ch1dn_impl<bi::ON_DEVICE,T1>::func(M1 U, V1 a, V2 b)
    throw (CholeskyException) {
  BI_ERROR_MSG(false, "ch1dn not implemented for device");
}

template<class T1>
template<class V1>
void bi::scal_impl<bi::ON_DEVICE,T1>::func(T1 alpha, V1 x) {
  CUBLAS_CHECKED_CALL(
      cublas_scal < T1
          > ::func(bi_omp_cublas_handle, x.size(), &alpha, x.buf(), x.inc()));
}

template<class T1>
template<class V1, class V2>
inline T1 bi::dot_impl<bi::ON_DEVICE,T1>::func(const V1 a, const V2 b) {
  T1 result;
  CUBLAS_CHECKED_CALL(
      cublas_dot < T1
          > ::func(bi_omp_cublas_handle, a.size(), a.buf(), a.inc(), b.buf(),
              b.inc(), &result));
  synchronize (bi_omp_cublas_handle);
  return result;
}

template<class T1>
template<class V1>
inline typename V1::size_type bi::iamax_impl<bi::ON_DEVICE,T1>::func(
    const V1 x) {
  typename V1::size_type result;
  CUBLAS_CHECKED_CALL(
      cublas_iamax < T1
          > ::func(bi_omp_cublas_handle, x.size(), x.buf(), x.inc(),
              &result));
  synchronize (bi_omp_cublas_handle);
  --result;  // to base zero
  return result;
}

template<class T1>
template<class V1, class V2>
inline void bi::axpy_impl<bi::ON_DEVICE,T1>::func(const T1 a, const V1 x,
    V2 y, const bool clear) {
  if (clear) {
    y.clear();
  }
  CUBLAS_CHECKED_CALL(
      cublas_axpy < T1
          > ::func(bi_omp_cublas_handle, y.size(), &a, x.buf(), x.inc(),
              y.buf(), y.inc()));
}

template<class T1>
template<class M1, class V1, class V2>
void bi::gemv_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A,
    const V1 x, const T1 beta, V2 y, const char transA) {
  CUBLAS_CHECKED_CALL(
      cublas_gemv < T1
          > ::func(bi_omp_cublas_handle, cublas_trans(transA), A.size1(),
              A.size2(), &alpha, A.buf(), A.lead(), x.buf(), x.inc(), &beta,
              y.buf(), y.inc()));
}

template<class T1>
template<class M1, class V1, class V2>
void bi::symv_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A,
    const V1 x, const T1 beta, V2 y, const char uplo) {
  CUBLAS_CHECKED_CALL(
      cublas_symv < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo), A.size1(), &alpha,
              A.buf(), A.lead(), x.buf(), x.inc(), &beta, y.buf(), y.inc()));
}

template<class T1>
template<class M1, class V1>
void bi::trmv_impl<bi::ON_DEVICE,T1>::func(const M1 A, V1 x, const char uplo,
    const char transA) {
  CUBLAS_CHECKED_CALL(
      cublas_trmv < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo),
              cublas_trans(transA), cublas_diag('N'), x.size(), A.buf(),
              A.lead(), x.buf(), x.inc()));
}

template<class T1>
template<class M1, class M2, class M3>
void bi::gemm_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A,
    const M2 X, const T1 beta, M3 Y, const char transA, const char transX) {
  host_matrix_reference<real>::size_type m =
      (transA == 'T') ? A.size2() : A.size1();
  BI_ASSERT(m == Y.size1());
  host_matrix_reference<real>::size_type n =
      (transX == 'T') ? X.size1() : X.size2();
  BI_ASSERT(n == Y.size2());
  host_matrix_reference<real>::size_type k =
      (transA == 'T') ? A.size1() : A.size2();

  CUBLAS_CHECKED_CALL(
      cublas_gemm < T1
          > ::func(bi_omp_cublas_handle, cublas_trans(transA),
              cublas_trans(transX), m, n, k, &alpha, A.buf(), A.lead(),
              X.buf(), X.lead(), &beta, Y.buf(), Y.lead()));
}

template<class T1>
template<class M1, class M2, class M3>
void bi::symm_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A,
    const M2 X, const T1 beta, M3 Y, const char side, const char uplo) {
  CUBLAS_CHECKED_CALL(
      cublas_symm < T1
          > ::func(bi_omp_cublas_handle, cublas_side(side), cublas_uplo(uplo),
              Y.size1(), Y.size2(), &alpha, A.buf(), A.lead(), X.buf(),
              X.lead(), &beta, Y.buf(), Y.lead()));
}

template<class T1>
template<class M1, class M2>
void bi::trmm_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A, M2 B,
    const char side, const char uplo, const char transA) {
  CUBLAS_CHECKED_CALL(
      cublas_trmm < T1
          > ::func(bi_omp_cublas_handle, cublas_side(side), cublas_uplo(uplo),
              cublas_trans(transA), cublas_diag('N'), B.size1(), B.size2(),
              &alpha, A.buf(), A.lead(), B.buf(), B.lead(), B.buf(),
              B.lead()));
  ///@todo Include different output matrix as option for CUBLAS
}

template<class T1>
template<class V1, class V2, class V3>
void bi::gdmv_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const V1 A,
    const V2 x, const T1 beta, V3 y) {
  CUBLAS_CHECKED_CALL(
      cublas_gbmv < T1
          > ::func(bi_omp_cublas_handle, cublas_trans('N'), A.size(),
              A.size(), 0, 0, &alpha, A.buf(), A.inc(), x.buf(), x.inc(),
              &beta, y.buf(), y.inc()));
  synchronize (bi_omp_cublas_handle);
}

template<class T1>
template<class V1, class V2, class M1>
void bi::ger_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const V1 x,
    const V2 y, M1 A, const bool clear) {
  CUBLAS_CHECKED_CALL(
      cublas_ger < T1
          > ::func(bi_omp_cublas_handle, A.size1(), A.size2(), &alpha,
              x.buf(), x.inc(), y.buf(), y.inc(), A.buf(), A.lead()));
}

template<class T1>
template<class V1, class M1>
void bi::syr_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const V1 x, M1 A,
    const char uplo, const bool clear) {
  CUBLAS_CHECKED_CALL(
      cublas_syr < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo), A.size1(), &alpha,
              x.buf(), x.inc(), A.buf(), A.lead()));
}

template<class T1>
template<class V1, class V2, class M1>
void bi::syr2_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const V1 x,
    const V2 y, M1 A, const char uplo, const bool clear) {
  if (clear) {
    A.clear();
  }
  CUBLAS_CHECKED_CALL(
      cublas_syr2 < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo), A.size1(), &alpha,
              x.buf(), x.inc(), y.buf(), y.inc(), A.buf(), A.lead()));
}

template<class T1>
template<class M1, class M2>
void bi::syrk_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A,
    const T1 beta, M2 C, const char uplo, const char trans) {
  typename M2::size_type k = (trans == 'T') ? A.size1() : A.size2();
  CUBLAS_CHECKED_CALL(
      cublas_syrk < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo),
              cublas_trans(trans), C.size1(), k, &alpha, A.buf(), A.lead(),
              &beta, C.buf(), C.lead()));
}

template<class T1>
template<class M1, class V1>
void bi::trsv_impl<bi::ON_DEVICE,T1>::func(const M1 A, V1 x, const char uplo,
    const char trans, const char diag) {
  CUBLAS_CHECKED_CALL(
      cublas_trsv < T1
          > ::func(bi_omp_cublas_handle, cublas_uplo(uplo),
              cublas_trans(trans), cublas_diag(diag), x.size(), A.buf(),
              A.lead(), x.buf(), x.inc()));
}

template<class T1>
template<class M1, class M2>
void bi::trsm_impl<bi::ON_DEVICE,T1>::func(const T1 alpha, const M1 A, M2 B,
    const char side, const char uplo, const char trans, const char diag) {
  CUBLAS_CHECKED_CALL(
      cublas_trsm < T1
          > ::func(bi_omp_cublas_handle, cublas_side(side), cublas_uplo(uplo),
              cublas_trans(trans), cublas_diag(diag), B.size1(), B.size2(),
              &alpha, A.buf(), A.lead(), B.buf(), B.lead()));
}

template<class T1>
template<class M1>
void bi::potrf_impl<bi::ON_DEVICE,T1>::func(M1 U, char uplo)
    throw (CholeskyException) {
  BI_ERROR_MSG(false, "Not implemented");
}

template<class T1>
template<class M1, class M2>
void bi::potrs_impl<bi::ON_DEVICE,T1>::func(const M1 U, M2 X, char uplo)
    throw (CholeskyException) {
  BI_ERROR_MSG(false, "Not implemented");
}

template<class T1>
template<class M1, class V1, class M2, class V2, class V3, class V4, class V5>
void bi::syevx_impl<bi::ON_DEVICE,T1>::func(char jobz, char range, char uplo,
    M1 A, typename M1::value_type vl, typename M1::value_type vu, int il,
    int iu, typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work,
    V3 rwork, V4 iwork, V5 ifail) throw (EigenException) {
  BI_ERROR_MSG(false, "Not implemented");
}

#endif
