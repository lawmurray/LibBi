/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_MULTIOPERATION_HPP
#define BI_HOST_MATH_MULTIOPERATION_HPP

namespace bi {
/**
 * @internal
 */
template<class T1>
struct multi_ch1dn_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(const int P, M1 U, V1 a, V2 b)
      throw (CholeskyException);
};

/**
 * @internal
 */
template<class T1>
struct multi_gemv_impl<ON_HOST,T1> {
  template<class M1, class V1, class V2>
  static void func(const int P, const T1 alpha, const M1 A, const V1 x,
      const T1 beta, V2 y, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct multi_gemm_impl<ON_HOST,T1> {
  template<class M1, class M2, class M3>
  static void func(const int P, const typename M1::value_type alpha,
      const M1 A, const M2 X, const typename M3::value_type beta, M3 Y,
      const char transA, const char transX);
};

/**
 * @internal
 */
template<class T1>
struct multi_trmm_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const int P, const typename M1::value_type alpha, const M1 A, M2 B,
      const char side, const char uplo, const char transA);
};

/**
 * @internal
 */
template<class T1>
struct multi_syrk_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const int P, const T1 alpha, const M1 As, const T1 beta,
      M2 Cs, const char uplo, const char trans);
};

/**
 * @internal
 */
template<class T1>
struct multi_trsv_impl<ON_HOST,T1> {
  template<class M1, class V1>
  static void func(const int P, const M1 A, V1 x, const char uplo,
      const char trans, const char diag);
};

/**
 * @internal
 */
template<class T1>
struct multi_trsm_impl<ON_HOST,T1> {
  template<class M1, class M2>
  static void func(const int P, const T1 alpha, const M1 A, M2 B,
      const char side, const char uplo, const char trans, const char diag);
};

/**
 * @internal
 */
template<class T1>
struct multi_potrf_impl<ON_HOST,T1> {
  template<class M1>
  static void func(const int P, M1 Us, char uplo) throw (CholeskyException);
};

}

#include "operation.hpp"
#include "cblas.hpp"
#include "lapack.hpp"
#include "qrupdate.hpp"

template<class T1>
template<class M1, class V1, class V2>
void bi::multi_ch1dn_impl<bi::ON_HOST,T1>::func(const int P, M1 Us, V1 as, V2 bs)
    throw (CholeskyException) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type U(Us.size1()/P, Us.size2());
    typename sim_temp_vector<V1>::type a(as.size()/P);
    typename sim_temp_vector<V2>::type b(as.size()/P);
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, Us, p, U);
      multi_get_vector(P, as, p, a);
      multi_get_vector(P, bs, p, b);

      ch1dn(U, a, b);

      multi_set_matrix(P, Us, p, U);
      multi_set_vector(P, as, p, a);
      multi_set_vector(P, bs, p, b);
    }
  }
}

template<class T1>
template<class M1, class V1, class V2>
void bi::multi_gemv_impl<bi::ON_HOST,T1>::func(const int P, const T1 alpha,
    const M1 As, const V1 xs, const T1 beta, V2 ys, const char transA) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_vector<V1>::type x(xs.size()/P);
    typename sim_temp_vector<V2>::type y(ys.size()/P);
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_vector(P, xs, p, x);
      multi_get_vector(P, ys, p, y);

      gemv(alpha, A, x, beta, y, transA);

      multi_set_vector(P, ys, p, y);
    }
  }
}

template<class T1>
template<class M1, class M2, class M3>
void bi::multi_gemm_impl<bi::ON_HOST,T1>::func(const int P,
    const typename M1::value_type alpha, const M1 As, const M2 Xs,
    const typename M3::value_type beta, M3 Ys, const char transA,
    const char transX) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_matrix<M2>::type X(Xs.size1()/P, Xs.size2());
    typename sim_temp_matrix<M2>::type Y(Ys.size1()/P, Ys.size2());
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_matrix(P, Xs, p, X);
      multi_get_matrix(P, Ys, p, Y);

      gemm(alpha, A, X, beta, Y, transA, transX);

      multi_set_matrix(P, Ys, p, Y);
    }
  }
};

template<class T1>
template<class M1, class M2>
void bi::multi_trmm_impl<bi::ON_HOST,T1>::func(const int P,
    const typename M1::value_type alpha, const M1 As, M2 Bs, const char side,
    const char uplo, const char transA) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_matrix<M2>::type B(Bs.size1()/P, Bs.size2());
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_matrix(P, Bs, p, B);

      trmm(alpha, A, B, side, uplo, transA);

      multi_set_matrix(P, Bs, p, B);
    }
  }
};

template<class T1>
template<class M1, class M2>
void bi::multi_syrk_impl<bi::ON_HOST,T1>::func(const int P, const T1 alpha,
    const M1 As, const T1 beta, M2 Cs, const char uplo, const char trans) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_matrix<M2>::type C(Cs.size1()/P, Cs.size2());
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_matrix(P, Cs, p, C);

      syrk(alpha, A, beta, C, uplo, trans);

      multi_set_matrix(P, Cs, p, C);
    }
  }
}

template<class T1>
template<class M1, class V1>
void bi::multi_trsv_impl<bi::ON_HOST,T1>::func(const int P, const M1 As, V1 xs,
    const char uplo, const char trans, const char diag) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_vector<V1>::type x(xs.size()/P);
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_vector(P, xs, p, x);

      trsv(A, x, uplo, trans, diag);

      multi_set_vector(P, xs, p, x);
    }
  }
}

template<class T1>
template<class M1, class M2>
void bi::multi_trsm_impl<bi::ON_HOST,T1>::func(const int P,
    const T1 alpha, const M1 As, M2 Xs, const char side,
    const char uplo, const char trans, const char diag) {
  #pragma omp parallel
  {
    typename sim_temp_matrix<M1>::type A(As.size1()/P, As.size2());
    typename sim_temp_matrix<M2>::type X(Xs.size1()/P, Xs.size2());
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, As, p, A);
      multi_get_matrix(P, Xs, p, X);

      trsm(alpha, A, X, side, uplo, trans, diag);

      multi_set_matrix(P, Xs, p, X);
    }
  }
}

template<class T1>
template<class M1>
void bi::multi_potrf_impl<bi::ON_HOST,T1>::func(const int P, M1 Us,
    char uplo) throw (CholeskyException) {
  int nerrs = 0;

  #pragma omp parallel reduction(+:nerrs)
  {
    typename sim_temp_matrix<M1>::type U(Us.size1()/P, Us.size2());
    int p;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      multi_get_matrix(P, Us, p, U);

      try {
        potrf(U, uplo);
      } catch (CholeskyException e) {
        ++nerrs;
      }

      multi_set_matrix(P, Us, p, U);
    }
  }

  if (nerrs > 0) {
    throw CholeskyException(0);
  }
}

#endif
