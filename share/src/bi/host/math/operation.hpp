/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_OPERATION_HPP
#define BI_HOST_MATH_OPERATION_HPP

namespace bi {
/**
 * @internal
 */
template<class T1>
struct ch1up_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b);
};

/**
 * @internal
 */
template<class T1>
struct ch1dn_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(M1 U, V1 a, V2 b) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct scal_impl<ON_HOST,T1> {
  template<class V1>
  static void func(T1 alpha, V1 x);
};

/**
 * @internal
 */
template<class T1>
struct dot_impl<ON_HOST,T1> {
  template<class V1, class V2>
  static T1 func(const V1 a, const V2 b);
};

/**
 * @internal
 */
template<class T1>
struct iamax_impl<ON_HOST,T1> {
  template<class V1>
  static typename V1::size_type func(const V1 x);
};

/**
 * @internal
 */
template<class T1>
struct axpy_impl<ON_HOST,T1> {
  template<class V1, class V2>
  static void func(const T1 a, const V1 x, V2 y, const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct gemv_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct symv_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(const T1 alpha, const M1 A, const V1 x, const T1 beta,
      V2 y, const char uplo);
};

/**
 * @internal
 */
template<class T1>
struct trmv_impl<ON_HOST,T1> {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct gdmv_impl<ON_HOST,T1> {
  template<class V1, class V2, class V3>
  static void func(const T1 alpha, const V1 A, const V2 x, const T1 beta,
      V3 y);
};

/**
 * @internal
 */
template<class T1>
struct gemm_impl<ON_HOST,T1> {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char transA, const char transX);
};

/**
 * @internal
 */
template<class T1>
struct symm_impl<ON_HOST,T1> {
  template<class M1, class M2, class M3>
  static void func(const T1 alpha, const M1 A, const M2 X, const T1 beta,
      M3 Y, const char side, const char uplo);
};

/**
 * @internal
 */
template<class T1>
struct trmm_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct ger_impl<ON_HOST,T1> {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syr_impl<ON_HOST,T1> {
  template<class V1, class M1>
  static void func(const T1 alpha, const V1 x, M1 A, const char uplo,
      const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syr2_impl<ON_HOST,T1> {
  template<class V1, class V2, class M1>
  static void func(const T1 alpha, const V1 x, const V2 y, M1 A,
      const char uplo, const bool clear);
};

/**
 * @internal
 */
template<class T1>
struct syrk_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, const T1 beta, M2 C,
      const char uplo, const char trans);
};

/**
 * @internal
 */
template<class T1>
struct trsv_impl<ON_HOST,T1> {
  template<class M1, class V1>
  static void func(const M1 A, V1 x, const char uplo, const char trans,
      const char diag);
};

/**
 * @internal
 */
template<class T1>
struct trsm_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const T1 alpha, const M1 A, M2 B, const char side,
      const char uplo, const char trans, const char diag);
};

/**
 * @internal
 */
template<class T1>
struct potrf_impl<ON_HOST,T1> {
  template<class M1>
  static void func(M1 U, char uplo) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct potrs_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const M1 U, M2 X, char uplo) throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct syevx_impl<ON_HOST,T1> {
  template<class M1, class V1, class M2, class V2, class V3, class V4,
      class V5>
  static void func(char jobz, char range, char uplo, M1 A,
      typename M1::value_type vl, typename M1::value_type vu, int il, int iu,
      typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work, V3 rwork,
      V4 iwork, V5 ifail) throw (EigenException);
};

}

#include "cblas.hpp"
#include "lapack.hpp"
#include "qrupdate.hpp"

template<class T1>
template<class M1, class V1, class V2>
void bi::ch1up_impl<bi::ON_HOST,T1>::func(M1 U, V1 a, V2 b) {
  int n = a.size();
  int ld = U.lead();
  qrupdate_ch1up < T1 > ::func(&n, U.buf(), &ld, a.buf(), b.buf());
}

template<class T1>
template<class M1, class V1, class V2>
void bi::ch1dn_impl<bi::ON_HOST,T1>::func(M1 U, V1 a, V2 b)
    throw (CholeskyException) {
  int n = a.size();
  int ld = U.lead();
  int info;
  qrupdate_ch1dn < T1 > ::func(&n, U.buf(), &ld, a.buf(), b.buf(), &info);
  if (info != 0) {
    throw CholeskyException(info);
  }
}

template<class T1>
template<class V1>
void bi::scal_impl<bi::ON_HOST,T1>::func(T1 alpha, V1 x) {
  cblas_scal < T1 > ::func(x.size(), alpha, x.buf(), x.inc());
}

template<class T1>
template<class V1, class V2>
inline T1 bi::dot_impl<bi::ON_HOST,T1>::func(const V1 a, const V2 b) {
  return cblas_dot < T1 > ::func(a.size(), a.buf(), a.inc(), b.buf(), b.inc());
}

template<class T1>
template<class V1>
inline typename V1::size_type bi::iamax_impl<bi::ON_HOST,T1>::func(
    const V1 x) {
  return cblas_iamax < T1 > ::func(x.size(), x.buf(), x.inc()) - 1;  // -1 so as to get zero base
}

template<class T1>
template<class V1, class V2>
inline void bi::axpy_impl<bi::ON_HOST,T1>::func(const T1 a, const V1 x, V2 y,
    const bool clear) {
  if (clear) {
    y.clear();
  }
  cblas_axpy < T1 > ::func(y.size(), a, x.buf(), x.inc(), y.buf(), y.inc());
}

template<class T1>
template<class M1, class V1, class V2>
void bi::gemv_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A,
    const V1 x, const T1 beta, V2 y, const char transA) {
  cblas_gemv < T1
      > ::func(CblasColMajor, cblas_trans(transA), A.size1(), A.size2(),
          alpha, A.buf(), A.lead(), x.buf(), x.inc(), beta, y.buf(), y.inc());
}

template<class T1>
template<class M1, class V1, class V2>
void bi::symv_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A,
    const V1 x, const T1 beta, V2 y, const char uplo) {
  cblas_symv < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), A.size1(), alpha, A.buf(),
          A.lead(), x.buf(), x.inc(), beta, y.buf(), y.inc());
}

template<class T1>
template<class M1, class V1>
void bi::trmv_impl<bi::ON_HOST,T1>::func(const M1 A, V1 x, const char uplo,
    const char transA) {
  cblas_trmv < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), cblas_trans(transA),
          cblas_diag('N'), x.size(), A.buf(), A.lead(), x.buf(), x.inc());
}

template<class T1>
template<class M1, class M2, class M3>
void bi::gemm_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A,
    const M2 X, const T1 beta, M3 Y, const char transA, const char transX) {
  host_matrix_reference<real>::size_type m =
      (transA == 'T') ? A.size2() : A.size1();
  BI_ASSERT(m == Y.size1());
  host_matrix_reference<real>::size_type n =
      (transX == 'T') ? X.size1() : X.size2();
  BI_ASSERT(n == Y.size2());
  host_matrix_reference<real>::size_type k =
      (transA == 'T') ? A.size1() : A.size2();

  cblas_gemm < T1
      > ::func(CblasColMajor, cblas_trans(transA), cblas_trans(transX), m, n,
          k, alpha, A.buf(), A.lead(), X.buf(), X.lead(), beta, Y.buf(),
          Y.lead());
}

template<class T1>
template<class M1, class M2, class M3>
void bi::symm_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A,
    const M2 X, const T1 beta, M3 Y, const char side, const char uplo) {
  cblas_symm < T1
      > ::func(CblasColMajor, cblas_side(side), cblas_uplo(uplo), Y.size1(),
          Y.size2(), alpha, A.buf(), A.lead(), X.buf(), X.lead(), beta,
          Y.buf(), Y.lead());
}

template<class T1>
template<class M1, class M2>
void bi::trmm_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A, M2 B,
    const char side, const char uplo, const char transA) {
  cblas_trmm < T1
      > ::func(CblasColMajor, cblas_side(side), cblas_uplo(uplo),
          cblas_trans(transA), cblas_diag('N'), B.size1(), B.size2(), alpha,
          A.buf(), A.lead(), B.buf(), B.lead());
}

template<class T1>
template<class V1, class V2, class V3>
void bi::gdmv_impl<bi::ON_HOST,T1>::func(const T1 alpha, const V1 A,
    const V2 x, const T1 beta, V3 y) {
  cblas_gbmv < T1
      > ::func(CblasColMajor, cblas_trans('N'), A.size(), A.size(), 0, 0,
          alpha, A.buf(), A.inc(), x.buf(), x.inc(), beta, y.buf(), y.inc());
}

template<class T1>
template<class V1, class V2, class M1>
void bi::ger_impl<bi::ON_HOST,T1>::func(const T1 alpha, const V1 x,
    const V2 y, M1 A, const bool clear) {
  cblas_ger < T1
      > ::func(CblasColMajor, A.size1(), A.size2(), alpha, x.buf(), x.inc(),
          y.buf(), y.inc(), A.buf(), A.lead());
}

template<class T1>
template<class V1, class M1>
void bi::syr_impl<bi::ON_HOST,T1>::func(const T1 alpha, const V1 x, M1 A,
    const char uplo, const bool clear) {
  cblas_syr < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), A.size1(), alpha, x.buf(),
          x.inc(), A.buf(), A.lead());
}

template<class T1>
template<class V1, class V2, class M1>
void bi::syr2_impl<bi::ON_HOST,T1>::func(const T1 alpha, const V1 x,
    const V2 y, M1 A, const char uplo, const bool clear) {
  if (clear) {
    A.clear();
  }
  cblas_syr2 < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), A.size1(), alpha, x.buf(),
          x.inc(), y.buf(), y.inc(), A.buf(), A.lead());
}

template<class T1>
template<class M1, class M2>
void bi::syrk_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A,
    const T1 beta, M2 C, const char uplo, const char trans) {
  typename M2::size_type k = (trans == 'T') ? A.size1() : A.size2();
  cblas_syrk < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), cblas_trans(trans), C.size1(),
          k, alpha, A.buf(), A.lead(), beta, C.buf(), C.lead());
}

template<class T1>
template<class M1, class V1>
void bi::trsv_impl<bi::ON_HOST,T1>::func(const M1 A, V1 x, const char uplo,
    const char trans, const char diag) {
  cblas_trsv < T1
      > ::func(CblasColMajor, cblas_uplo(uplo), cblas_trans(trans),
          cblas_diag(diag), x.size(), A.buf(), A.lead(), x.buf(), x.inc());
}

template<class T1>
template<class M1, class M2>
void bi::trsm_impl<bi::ON_HOST,T1>::func(const T1 alpha, const M1 A, M2 B,
    const char side, const char uplo, const char trans, const char diag) {
  cblas_trsm < T1
      > ::func(CblasColMajor, cblas_side(side), cblas_uplo(uplo),
          cblas_trans(trans), cblas_diag(diag), B.size1(), B.size2(), alpha,
          A.buf(), A.lead(), B.buf(), B.lead());
}

template<class T1>
template<class M1>
void bi::potrf_impl<bi::ON_HOST,T1>::func(M1 U, char uplo)
    throw (CholeskyException) {
  int info;
  int N = U.size1();
  int ld = U.lead();

  lapack_potrf < T1 > ::func(&uplo, &N, U.buf(), &ld, &info);
  if (info != 0) {
    throw CholeskyException(info);
  }
}

template<class T1>
template<class M1, class M2>
void bi::potrs_impl<bi::ON_HOST,T1>::func(const M1 U, M2 X, char uplo)
    throw (CholeskyException) {
  int info;
  int N = U.size1();
  int M = X.size2();
  int ldL = U.lead();
  int ldX = X.lead();

  lapack_potrs < T1
      > ::func(&uplo, &N, &M, U.buf(), &ldL, X.buf(), &ldX, &info);
  if (info != 0) {
    throw CholeskyException(info);
  }
}

template<class T1>
template<class M1, class V1, class M2, class V2, class V3, class V4, class V5>
void bi::syevx_impl<bi::ON_HOST,T1>::func(char jobz, char range, char uplo,
    M1 A, typename M1::value_type vl, typename M1::value_type vu, int il,
    int iu, typename M1::value_type abstol, int* m, V1 w, M2 Z, V2 work,
    V3 rwork, V4 iwork, V5 ifail) throw (EigenException) {
  int info;
  int N = A.size1();
  int ldA = A.lead();
  int ldZ = Z.lead();
  int lwork = work.size();

  lapack_syevx < T1
      > ::func(&jobz, &range, &uplo, &N, A.buf(), &ldA, &vl, &vu, &il, &iu,
          &abstol, m, w.buf(), Z.buf(), &ldZ, work.buf(), &lwork, iwork.buf(),
          ifail.buf(), &info);
  if (info != 0) {
    throw EigenException(info);
  }
}

#endif
