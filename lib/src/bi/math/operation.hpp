/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * @todo Check synchronize() only used where necessary.
 */
#ifndef BI_CUDA_MATH_OPERATION_HPP
#define BI_CUDA_MATH_OPERATION_HPP

namespace bi {
/**
 * @name Basic operations
 */
//@{
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
 * Set the columns of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Sets each column of @p A to @p x.
 */
template<class M1, class V1>
void set_columns(M1 A, const V1 x);

/**
 * Set the rows of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Sets each row of @p A to @p x.
 */
template<class M1, class V1>
void set_rows(M1 A, const V1 x);

/**
 * Add a vector to the columns of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Adds @p x to each column of @p A.
 */
template<class M1, class V1>
void add_columns(M1 A, const V1 x);

/**
 * Add a vector to the rows of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Adds @p x to each row of @p A.
 */
template<class M1, class V1>
void add_rows(M1 A, const V1 x);

/**
 * Subtract a vector from the columns of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Subtracts @p x from each column of @p A.
 */
template<class M1, class V1>
void sub_columns(M1 A, const V1 x);

/**
 * Subtract a vector from the rows of a matrix.
 *
 * @ingroup math_op
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Subtracts @p x from each row of @p A.
 */
template<class M1, class V1>
void sub_rows(M1 A, const V1 x);

/**
 * Compute the dot product of each column of a matrix with itself.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void dot_columns(const M1 X, V1 y);

/**
 * Compute the dot product of each row of a matrix with itself.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void dot_rows(const M1 X, V1 y);

/**
 * Sum the columns of a matrix.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void sum_columns(const M1 X, V1 y);

/**
 * Sum the rows of a matrix.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void sum_rows(const M1 X, V1 y);

//@}
/**
 * @name BLAS/LAPACK operations
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
 * Scalar multiply and vector add.
 *
 * @ingroup math_op
 */
template<class T1, class V1, class V2>
void axpy(const T1 a, const V1 x, V2 y, const bool clear = false);

/**
 * Matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class V1, class T2, class V2>
void gemv(const T1 alpha, const M1 A, const V1 x, const T2 beta,
    V2 y, const char transA = 'N');

/**
 * Symmetric matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class V1, class T2, class V2>
void symv(const T1 alpha, const M1 A, const V1 x, const T2 beta,
    V2 y, const char uplo = 'L');

/**
 * Triangular matrix-vector multiply.
 *
 * @ingroup math_op
 */
template<class M1, class V1>
void trmv(const M1 A, V1 x, const char uplo = 'L', const char transA = 'N');

/**
 * Diagonal matrix-vector multiply.
 *
 * @ingroup math_op
 *
 * Uses @c gbmv internally, with single leading diagonal band.
 */
template<class T1, class V1, class V2, class T2, class V3>
void gdmv(const T1 alpha, const V1 A, const V2 x, const T2 beta, V3 y);

/**
 * Scalar multiply and matrix add.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class M2>
void matrix_axpy(const T1 a, const M1 X, M2 Y, const bool clear = false);

/**
 * Matrix scale.
 *
 * @ingroup math_op
 */
template<class M1>
void matrix_scal(typename M1::value_type alpha, M1 X);

/**
 * Matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class M2, class T2, class M3>
void gemm(const T1 alpha, const M1 A, const M2 X, const T2 beta,
    M3 Y, const char transA = 'N', const char transX = 'N');

/**
 * Symmetric matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class M2, class T2, class M3>
void symm(const T1 alpha, const M1 A, const M2 X, const T2 beta, M3 Y,
    const char side = 'L', const char uplo = 'L');

/**
 * Triangular matrix-matrix multiply.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class M2>
void trmm(const T1 alpha, const M1 A, M2 B, const char side = 'L',
    const char uplo = 'L', const char transA = 'N');

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
template<class T1, class V1, class M1, class T2, class M2>
void gdmm(const T1 alpha, const V1 A, const M1 X, const T2 beta, M2 Y,
    const char side = 'L');

/**
 * Vector outer product and matrix add.
 *
 * @ingroup math_op
 */
template<class T1, class V1, class V2, class M1>
void ger(const T1 alpha, const V1 x, const V2 y, M1 A,
    const bool clear = false);

/**
 * Symmetric vector outer product and matrix add.
 *
 * @ingroup math_op
 */
template<class T1, class V1, class M1>
void syr(const T1 alpha, const V1 x, M1 A, const char uplo = 'L',
    const bool clear = false);

/**
 * Symmetric matrix rank-2 update.
 *
 * @ingroup math_op
 */
template<class T1, class V1, class V2, class M1>
void syr2(const T1 alpha, const V1 x, const V2 y, M1 A,
    const char uplo = 'L', const bool clear = false);

/**
 * Matrix rank-k update.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class T2, class M2>
void syrk(const T1 alpha, const M1 A, const T2 beta, M2 C,
    const char uplo = 'L', const char trans = 'N');

/**
 * Symmetric matrix Cholesky decomposition.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void potrf(const M1 A, M2 L, char uplo = 'L');

/**
 * Symmetric positive definite linear system solve.
 *
 * @ingroup math_op
 */
template<class M1, class M2>
void potrs(const M1 L, M2 X, char uplo = 'L');

/**
 * Triangular linear system solve.
 *
 * @ingroup math_op
 */
template<class T1, class M1, class M2>
void trsm(const T1 alpha, const M1 A, M2 B, const char side = 'L',
    const char uplo = 'L', const char trans = 'N', const char diag = 'N');
//@}

}

#include "view.hpp"
#include "cblas.hpp"
#include "clapack.hpp"
#include "temp_vector.hpp"
#include "temp_matrix.hpp"
#include "primitive.hpp"
#include "../cuda/cuda.hpp"
#include "../cuda/math/cublas.hpp"
#include "../cuda/math/magma.hpp"
#include "../cuda/math/temp_vector.hpp"
#include "../cuda/math/temp_matrix.hpp"
#include "../typelist/equals.hpp"
#include "../misc/repeated_range.hpp"
#include "../misc/stuttered_range.hpp"

template<class M1>
inline void bi::ident(M1 A) {
  BOOST_AUTO(d, diagonal(A));
  A.clear();
  bi::fill(d.begin(), d.end(), static_cast<typename M1::value_type>(1.0));
}

template<class M1, class M2>
inline void bi::transpose(const M1 A, M2 B) {
  /* pre-condition */
  assert (A.size1() == B.size2() && A.size2() == B.size1());

  int i;
  for (i = 0; i < A.size1(); ++i) {
    column(B.ref(),i) = row(A.ref(),i);
  }
}

template<class M1, class V1>
inline void bi::set_columns(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size1() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(repeated, make_repeated_range(x1->begin(), x1->end(), A.size2()));
  bi::copy(repeated.begin(), repeated.end(), A.begin());
  synchronize();
  delete x1;
}

template<class M1, class V1>
inline void bi::set_rows(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size2() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(stuttered, make_stuttered_range(x1->begin(), x1->end(), A.size1()));
  bi::copy(stuttered.begin(), stuttered.end(), A.begin());
  synchronize();
  delete x1;
}

template<class M1, class V1>
inline void bi::add_columns(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size1() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(repeated, make_repeated_range(x1->begin(), x1->end(), A.size2()));
  thrust::transform(A.begin(), A.end(), repeated.begin(), A.begin(), thrust::plus<typename M1::value_type>());
  synchronize();
  delete x1;
}

template<class M1, class V1>
inline void bi::add_rows(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size2() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(stuttered, make_stuttered_range(x1->begin(), x1->end(), A.size1()));
  thrust::transform(A.begin(), A.end(), stuttered.begin(), A.begin(), thrust::plus<typename M1::value_type>());
  synchronize();
  delete x1;
}

template<class M1, class V1>
inline void bi::sub_columns(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size1() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(repeated, make_repeated_range(x1->begin(), x1->end(), A.size2()));

  thrust::transform(A.begin(), A.end(), repeated.begin(), A.begin(), thrust::minus<typename M1::value_type>());
  synchronize();
  delete x1;
}

template<class M1, class V1>
inline void bi::sub_rows(M1 A, const V1 x) {
  /* pre-condition */
  assert (A.size2() == x.size());

  BOOST_AUTO(x1, map_vector(A, x));
  BOOST_AUTO(stuttered, make_stuttered_range(x1->begin(), x1->end(), A.size1()));
  thrust::transform(A.begin(), A.end(), stuttered.begin(), A.begin(), thrust::minus<typename M1::value_type>());
  synchronize();
  delete x1;
}

template<class V1>
inline void bi::scal(typename V1::value_type alpha, V1 x) {
  typedef typename V1::value_type T1;

  if (V1::on_device) {
    cublas_scal<T1>::func(x.size(), alpha, x.buf(), x.inc());
  } else {
    cblas_scal<T1>::func(x.size(), alpha, x.buf(), x.inc());
  }
}

template<class V1, class V2>
inline typename V1::value_type bi::dot(const V1 a, const V2 b) {
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;

  /* pre-conditions */
  assert (a.size() == b.size());
  assert ((equals<T1,T2>::value));
  assert ((equals<T1,float>::value || equals<T1,double>::value));

  typename V1::value_type result;
  if (V1::on_device) {
    BOOST_AUTO(a1, gpu_map_vector(a));
    BOOST_AUTO(b1, gpu_map_vector(b));
    result = cublas_dot<T1>::func(a1->size(), a1->buf(), a1->inc(), b1->buf(), b1->inc());
    synchronize();
    CUBLAS_CHECK;
    delete a1;
    delete b1;
  } else {
    BOOST_AUTO(a1, host_map_vector(a));
    BOOST_AUTO(b1, host_map_vector(b));
    if (V1::on_device || V2::on_device) {
      synchronize();
    }
    result = cblas_dot<T1>::func(a1->size(), a1->buf(), a1->inc(), b1->buf(), b1->inc());
    delete a1;
    delete b1;
  }
  return result;
}

template<class V1>
inline typename V1::value_type bi::dot(const V1 a) {
  typedef typename V1::value_type T1;

  /* pre-conditions */
  assert ((equals<T1,float>::value || equals<T1,double>::value));

  T1 result;
  if (V1::on_device) {
    result = cublas_dot<T1>::func(a.size(), a.buf(), a.inc(), a.buf(), a.inc());
  } else {
    result = cblas_dot<T1>::func(a.size(), a.buf(), a.inc(), a.buf(), a.inc());
  }
  return result;
}

template<class V1>
inline typename V1::size_type bi::iamax(const V1 x) {
  typedef typename V1::value_type T1;

  if (V1::on_device) {
    return cublas_iamax<T1>::func(x.size(), x.buf(), x.inc()) - 1;
  } else {
    return cblas_iamax<T1>::func(x.size(), x.buf(), x.inc()) - 1;
  }
}

template<class T1, class V1, class V2>
inline void bi::axpy(const T1 a, const V1 x, V2 y, const bool clear) {
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;

  /* pre-conditions */
  assert (x.size() == y.size());
  //BOOST_STATIC_ASSERT((equals<T2,T3>::value));

  if (clear) {
    y.clear();
  }
  if (V2::on_device) {
    BOOST_AUTO(x1, gpu_map_vector(x));
    cublas_axpy<T3>::func(y.size(), a, x1->buf(), x1->inc(), y.buf(), y.inc());
    synchronize();
    CUBLAS_CHECK;
    delete x1;
  } else {
    BOOST_AUTO(x1, host_map_vector(x));
    if (V1::on_device) {
      synchronize();
    }

    cblas_axpy<T2>::func(y.size(), a, x1->buf(), x1->inc(), y.buf(), y.inc());
    delete x1;
  }
}

template<class T1, class M1, class V1, class T2, class V2>
void bi::gemv(const T1 alpha, const M1 A, const V1 x, const T2 beta,
    V2 y, const char transA) {
  typedef typename M1::value_type T3;
  typedef typename V1::value_type T4;
  typedef typename V2::value_type T5;

  /* pre-conditions */
  assert (transA == 'N' || transA == 'T');
  assert (transA != 'N' || (A.size2() == x.size() && A.size1() == y.size()));
  assert (transA != 'T' || (A.size1() == x.size() && A.size2() == y.size()));
  assert ((equals<T3,T4>::value));
  assert ((equals<T4,T5>::value));

  if (V2::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    BOOST_AUTO(x1, gpu_map_vector(x));
    cublas_gemv<T5>::func(transA, A1->size1(), A1->size2(), alpha, A1->buf(),
        A1->lead(), x1->buf(), x1->inc(), beta, y.buf(), y.inc());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
    delete x1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    BOOST_AUTO(x1, host_map_vector(x));
    if (M1::on_device || V1::on_device) {
      synchronize();
    }
    cblas_gemv<T5>::func(CblasColMajor, cblas_trans(transA), A1->size1(),
        A1->size2(), alpha, A1->buf(), A1->lead(), x1->buf(), x1->inc(),
        beta, y.buf(), y.inc());
    delete A1;
    delete x1;
  }
}

template<class T1, class M1, class V1, class T2, class V2>
void bi::symv(const T1 alpha, const M1 A, const V1 x, const T2 beta,
    V2 y, const char uplo = 'N') {
  typedef typename M1::value_type T3;
  typedef typename V1::value_type T4;
  typedef typename V2::value_type T5;

  /* pre-conditions */
  assert (uplo == 'U' || uplo == 'L');
  assert (A.size2() == x.size() && A.size1() == y.size());
  assert ((equals<T3,T4>::value));
  assert ((equals<T4,T5>::value));

  if (V2::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    BOOST_AUTO(x1, gpu_map_vector(x));
    cublas_symv<T5>::func(uplo, A1->size1(), alpha, A1->buf(),
        A1->lead(), x1->buf(), x1->inc(), beta, y.buf(), y.inc());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
    delete x1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    BOOST_AUTO(x1, host_map_vector(x));
    if (M1::on_device || V1::on_device) {
      synchronize();
    }
    cblas_symv<T5>::func(CblasColMajor, cblas_uplo(uplo), A1->size1(),
        alpha, A1->buf(), A1->lead(), x1->buf(), x1->inc(),
        beta, y.buf(), y.inc());
    delete A1;
    delete x1;
  }
}

template<class M1, class V1>
void bi::trmv(const M1 A, V1 x, const char uplo, const char transA) {
  typedef typename M1::value_type T1;
  typedef typename V1::value_type T2;

  /* pre-conditions */
  assert (uplo == 'U' || uplo == 'L');
  assert (transA == 'N' || transA == 'T');
  assert (transA != 'N' || A.size2() == x.size());
  assert (transA != 'T' || A.size1() == x.size());
  //BOOST_STATIC_ASSERT((equals<T1,T2>::value));

  if (V1::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    cublas_trmv<T2>::func(uplo, transA, 'N', x.size(), A1->buf(), A1->lead(),
        x.buf(), x.inc());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    if (M1::on_device) {
      synchronize();
    }
    cblas_trmv<T2>::func(CblasColMajor, cblas_uplo(uplo), cblas_trans(transA),
        cblas_diag('N'), x.size(), A1->buf(), A1->lead(), x.buf(), x.inc());
    delete A1;
  }
}

template<class M1, class V1>
void bi::dot_columns(const M1 X, V1 y) {
  /* pre-condition */
  assert (X.size2() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(tmp, temp_vector<V1>(X.size2())); // must output keys, although not needed
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size2(), X.size1()));
  BOOST_AUTO(transform, make_transform_iterator(X.begin(), square_functor<T1>()));

  reduce_by_key(keys.begin(), keys.end(), transform, tmp->begin(), y.begin());

  synchronize();
  delete tmp;
}

template<class M1, class V1>
void bi::dot_rows(const M1 X, V1 y) {
  /* pre-condition */
  assert (X.size1() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(tmp, temp_vector<V1>(X.size1())); // must output keys, although not needed
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size1(), X.size2()));
  BOOST_AUTO(transform, make_transform_iterator(X.row_begin(), square_functor<T1>()));

  reduce_by_key(keys.begin(), keys.end(), transform, tmp->begin(), y.begin());

  synchronize();
  delete tmp;
}

template<class M1, class V1>
void bi::sum_columns(const M1 X, V1 y) {
  /* pre-condition */
  assert (X.size1() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(tmp, temp_vector<V1>(X.size1())); // must output keys, although not needed
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size1(), X.size2()));

  reduce_by_key(keys.begin(), keys.end(), X.row_begin(), tmp->begin(), y.begin());

  synchronize();
  delete tmp;
}

template<class M1, class V1>
void bi::sum_rows(const M1 X, V1 y) {
  /* pre-condition */
  assert (X.size2() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(tmp, temp_vector<V1>(X.size2())); // must output keys, although not needed
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size2(), X.size1()));

  reduce_by_key(keys.begin(), keys.end(), X.begin(), tmp->begin(), y.begin());

  synchronize();
  delete tmp;
}

template<class T1, class M1, class M2>
void bi::matrix_axpy(const T1 a, const M1 X, M2 Y, const bool clear) {
  /* pre-conditions */
  assert (X.size1() == Y.size1() && X.size2() == Y.size2());

  if (X.size1() == X.lead() && Y.size1() == Y.lead()) {
    /* do as one vector axpy */
    axpy(a, vec(X), vec(Y), clear);
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < X.size2(); ++j) {
      axpy(a, column(X,j), column(Y,j), clear);
    }
  }
}

template<class M1>
inline void bi::matrix_scal(typename M1::value_type alpha, M1 X) {
  typedef typename M1::value_type T1;

  if (X.size1() == X.lead()) {
    /* do as one vector scal */
    scal(alpha, vec(X));
  } else {
    /* do column-by-column */
    int j;
    for (j = 0; j < X.size2(); ++j) {
      scal(alpha, column(X,j));
    }
  }
}

template<class T1, class M1, class M2, class T2, class M3>
void bi::gemm(const T1 alpha, const M1 A, const M2 X, const T2 beta,
    M3 Y, const char transA, const char transX) {
  typedef typename M1::value_type T3;
  typedef typename M2::value_type T4;
  typedef typename M3::value_type T5;

  /* pre-conditions */
  assert (transA == 'N' || transA == 'T');
  assert (transX == 'N' || transX == 'T');
  assert (!(transA == 'N' && transX == 'N') ||
      (A.size2() == X.size1() && A.size1() == Y.size1() && X.size2() == Y.size2()));
  assert (!(transA == 'T' && transX == 'T') ||
      (A.size1() == X.size2() && A.size2() == Y.size1() && X.size1() == Y.size2()));
  assert (!(transA == 'N' && transX == 'T') ||
      (A.size2() == X.size2() && A.size1() == Y.size1() && X.size1() == Y.size2()));
  assert (!(transA == 'T' && transX == 'N') ||
      (A.size1() == X.size1() && A.size2() == Y.size1() && X.size2() == Y.size2()));

  host_matrix_reference<real>::size_type m = (transA == 'T') ? A.size2() : A.size1();
  assert (m == Y.size1());
  host_matrix_reference<real>::size_type n = (transX == 'T') ? X.size1() : X.size2();
  assert (n == Y.size2());
  host_matrix_reference<real>::size_type k = (transA == 'T') ? A.size1() : A.size2();

  if (M3::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    BOOST_AUTO(X1, gpu_map_matrix(X));
    cublas_gemm<T5>::func(transA, transX, m, n, k, alpha, A1->buf(), A1->lead(),
        X1->buf(), X1->lead(), beta, Y.buf(), Y.lead());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
    delete X1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    BOOST_AUTO(X1, host_map_matrix(X));
    if (M1::on_device || M2::on_device) {
      synchronize();
    }
    cblas_gemm<T5>::func(CblasColMajor, cblas_trans(transA), cblas_trans(transX),
        m, n, k, alpha, A1->buf(), A1->lead(), X1->buf(), X1->lead(), beta,
        Y.buf(), Y.lead());
    delete A1;
    delete X1;
  }
}

template<class T1, class M1, class M2, class T2, class M3>
void bi::symm(const T1 alpha, const M1 A, const M2 X, const T2 beta, M3 Y,
    const char side, const char uplo) {
  typedef typename M1::value_type T3;
  typedef typename M2::value_type T4;
  typedef typename M3::value_type T5;

  /* pre-conditions */
  assert (side == 'L' || side == 'R');
  assert (uplo == 'U' || uplo == 'L');
  assert (!(side == 'L') ||
      (A.size2() == X.size1() && A.size1() == Y.size1() && X.size2() == Y.size2()));
  assert (!(side == 'R') ||
      (X.size2() == A.size1() && X.size1() == Y.size1() && A.size2() == Y.size2()));
  //BOOST_STATIC_ASSERT((equals<T3,T4>::value));
  //BOOST_STATIC_ASSERT((equals<T4,T5>::value));

  if (M3::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    BOOST_AUTO(X1, gpu_map_matrix(X));
    cublas_symm<T5>::func(side, uplo, Y.size1(), Y.size2(), alpha, A1->buf(),
        A1->lead(), X1->buf(), X1->lead(), beta, Y.buf(), Y.lead());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
    delete X1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    BOOST_AUTO(X1, host_map_matrix(X));
    if (M1::on_device || M2::on_device) {
      synchronize();
    }
    cblas_symm<T5>::func(CblasColMajor, cblas_side(side), cblas_uplo(uplo),
        Y.size1(), Y.size2(), alpha, A1->buf(), A1->lead(), X1->buf(),
        X1->lead(), beta, Y.buf(), Y.lead());
    delete A1;
    delete X1;
  }
}

template<class T1, class M1, class M2>
void bi::trmm(const T1 alpha, const M1 A, M2 B, const char side,
    const char uplo, const char transA) {
  typedef typename M1::value_type T2;
  typedef typename M2::value_type T3;

  /* pre-conditions */
  assert (side == 'L' || side == 'R');
  assert (uplo == 'U' || uplo == 'L');
  assert (transA == 'T' || transA == 'N');
  assert (!(transA == 'N' && side == 'L') || A.size2() == B.size1());
  assert (!(transA == 'T' && side == 'L') || A.size1() == B.size1());
  assert (!(transA == 'N' && side == 'R') || B.size2() == A.size1());
  assert (!(transA == 'T' && side == 'R') || B.size2() == A.size2());
  assert((equals<T2,T3>::value));

  if (M2::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    cublas_trmm<T3>::func(side, uplo, transA, 'N', B.size1(), B.size2(),
        alpha, A1->buf(), A1->lead(), B.buf(), B.lead());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    if (M1::on_device) {
      synchronize();
    }
    cblas_trmm<T3>::func(CblasColMajor, cblas_side(side), cblas_uplo(uplo),
        cblas_trans(transA), cblas_diag('N'), B.size1(), B.size2(), alpha,
        A1->buf(), A1->lead(), B.buf(), B.lead());
    delete A1;
  }
}

template<class T1, class V1, class V2, class T2, class V3>
void bi::gdmv(const T1 alpha, const V1 A, const V2 x, const T2 beta, V3 y) {
  typedef typename V3::value_type T3;

  /* pre-conditions */
  assert (A.size() == x.size());
  assert (x.size() == y.size());

  if (V3::on_device) {
    BOOST_AUTO(A1, gpu_map_vector(A));
    BOOST_AUTO(x1, gpu_map_vector(x));
    cublas_gbmv<T3>::func('N', A1->size(), A1->size(), 0, 0, alpha, A1->buf(),
        A1->inc(), x1->buf(), x1->inc(), beta, y.buf(), y.inc());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
    delete x1;
  } else {
    BOOST_AUTO(A1, host_map_vector(A));
    BOOST_AUTO(x1, host_map_vector(x));
    if (V1::on_device || V2::on_device) {
      synchronize();
    }
    cblas_gbmv<T3>::func(CblasColMajor, cblas_trans('N'), A1->size(),
        A1->size(), 0, 0, alpha, A1->buf(), A1->inc(), x1->buf(), x1->inc(),
        beta, y.buf(),
        y.inc());
    delete A1;
    delete x1;
  }
}

template<class T1, class V1, class M1, class T2, class M2>
void bi::gdmm(const T1 alpha, const V1 A, const M1 X, const T2 beta, M2 Y,
    const char side = 'L') {
  /* pre-conditions */
  assert (side == 'L' || side == 'R');
  assert (side != 'L' || (A.size() == Y.size1() && Y.size1() == A.size() && Y.size2() == X.size2()));
  assert (side != 'R' || (X.size2() == A.size() && Y.size1() == X.size1() && Y.size2() == A.size()));
  int j;

  /* while we can make this generic for both host and device types and let
   * subroutines handle temp vectors, more efficient is to create all the
   * temps here, such that any copying is performed only once, and as a few
   * large copies rather than many small copies to maximise bandwidth */
  if (M2::on_device) {
    BOOST_AUTO(X1, gpu_map_matrix(X));
    if (side == 'L') {
      /* gdmv on each column */
      BOOST_AUTO(A1, gpu_map_vector(A));
      for (j = 0; j < X1->size2(); ++j) {
        gdmv(alpha, *A1, column(*X1,j), beta, column(Y,j));
      }
      synchronize();
      CUBLAS_CHECK;
      delete A1;
    } else {
      /* scal and axpy on each column */
      BOOST_AUTO(A1, host_map_vector(A));
      if (V1::on_device) {
        synchronize();
      }
      if (beta == 0.0) {
        Y.clear(); // must do this to ensure 0*nan == 0, not 0*nan == nan
      } else {
        matrix_scal(beta, Y);
      }
      for (j = 0; j < X1->size2(); ++j) {
        axpy(alpha*(*A1)(j), column(*X1,j), column(Y,j));
      }
      synchronize();
      CUBLAS_CHECK;
      delete A1;
    }
    delete X1;
  } else {
    BOOST_AUTO(A1, host_map_vector(A));
    BOOST_AUTO(X1, host_map_matrix(X));
    if (V1::on_device || M1::on_device) {
      synchronize();
    }
    if (side == 'L') {
      /* gdmv on each column */
      for (j = 0; j < X1->size2(); ++j) {
        gdmv(alpha, *A1, column(*X1,j), beta, column(Y,j));
      }
    } else {
      /* scal and axpy on each column */
      if (beta == 0.0) {
        Y.clear(); // must do this to ensure 0*nan == 0, not 0*nan == nan
      } else {
        matrix_scal(beta, Y);
      }
      for (j = 0; j < X1->size2(); ++j) {
        axpy(alpha*(*A1)(j), column(*X1,j), column(Y,j));
      }
    }
    delete A1;
    delete X1;
  }
}

template<class T1, class V1, class V2, class M1>
void bi::ger(const T1 alpha, const V1 x, const V2 y, M1 A,
    const bool clear) {
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;
  typedef typename M1::value_type T4;

  /* pre-conditions */
  assert (x.size() == A.size1());
  assert (y.size() == A.size2());
  assert((equals<T2,T3>::value));
  assert((equals<T3,T4>::value));

  if (clear) {
    A.clear();
  }

  if (M1::on_device) {
    BOOST_AUTO(x1, gpu_map_vector(x));
    BOOST_AUTO(y1, gpu_map_vector(y));
    cublas_ger<T4>::func(A.size1(), A.size2(), alpha, x1->buf(), x1->inc(),
        y1->buf(), y1->inc(), A.buf(), A.lead());
    synchronize();
    CUBLAS_CHECK;
    delete x1;
    delete y1;
  } else {
    BOOST_AUTO(x1, host_map_vector(x));
    BOOST_AUTO(y1, host_map_vector(y));
    if (V1::on_device || V2::on_device) {
      synchronize();
    }
    cblas_ger<T4>::func(CblasColMajor, A.size1(), A.size2(), alpha, x1->buf(),
        x1->inc(), y1->buf(), y1->inc(), A.buf(), A.lead());
    delete x1;
    delete y1;
  }
}

template<class T1, class V1, class M1>
void bi::syr(const T1 alpha, const V1 x, M1 A, const char uplo,
    const bool clear) {
  typedef typename V1::value_type T2;
  typedef typename M1::value_type T3;

  /* pre-condition */
  assert (uplo == 'U' || uplo == 'L');
  assert (A.size1() == A.size2());
  assert (x.size() == A.size1());
  //BOOST_STATIC_ASSERT((equals<T2,T3>::value));

  if (clear) {
    A.clear();
  }

  if (M1::on_device) {
    BOOST_AUTO(x1, gpu_map_vector(x));
    cublas_syr<T3>::func(uplo, A.size1(), alpha, x1->buf(), x1->inc(),
        A.buf(), A.lead());
    synchronize();
    CUBLAS_CHECK;
    delete x1;
  } else {
    BOOST_AUTO(x1, host_map_vector(x));
    if (V1::on_device) {
      synchronize();
    }
    cblas_syr<T3>::func(CblasColMajor, cblas_uplo(uplo), A.size1(), alpha,
        x1->buf(), x1->inc(), A.buf(), A.lead());
    delete x1;
  }
}

template<class T1, class V1, class V2, class M1>
void bi::syr2(const T1 alpha, const V1 x, const V2 y, M1 A,
    const char uplo, const bool clear) {
  typedef typename V1::value_type T2;
  typedef typename V2::value_type T3;
  typedef typename M1::value_type T4;

  /* pre-conditions */
  assert (uplo == 'U' || uplo == 'L');
  assert ((equals<T2,T3>::value));
  assert ((equals<T3,T4>::value));

  if (clear) {
    A.clear();
  }
  if (M1::on_device) {
    BOOST_AUTO(x1, gpu_map_vector(x));
    BOOST_AUTO(y1, gpu_map_vector(y));
    cublas_syr2<T4>::func(uplo, A.size1(), alpha, x1->buf(), x1->inc(),
        y1->buf(), y1->inc(), A.buf(), A.lead());
    synchronize();
    CUBLAS_CHECK;
    delete x1;
    delete y1;
  } else {
    BOOST_AUTO(x1, host_map_vector(x));
    BOOST_AUTO(y1, host_map_vector(y));
    if (V1::on_device || V2::on_device) {
      synchronize();
    }
    cblas_syr2<T4>::func(CblasColMajor, cblas_uplo(uplo), A.size1(), alpha,
        x1->buf(), x1->inc(), y1->buf(), y1->inc(), A.buf(), A.lead());
    delete x1;
    delete y1;
  }
}

template<class T1, class M1, class T2, class M2>
void bi::syrk(const T1 alpha, const M1 A, const T2 beta, M2 C,
    const char uplo, const char trans) {
  typedef typename M1::value_type T3;
  typedef typename M2::value_type T4;

  /* pre-conditions */
  assert (trans == 'N' || trans == 'T');
  assert (uplo == 'U' || uplo == 'L');
  assert (C.size1() == C.size2());
  assert (trans != 'N' || A.size1() == C.size1());
  assert (trans != 'T' || A.size2() == C.size1());
  //BOOST_STATIC_ASSERT((equals<T3,T4>::value));

  typename M2::size_type k = (trans == 'T') ? A.size1() : A.size2();

  if (M2::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    cublas_syrk<T4>::func(uplo, trans, C.size1(), k, alpha, A1->buf(),
        A1->lead(), beta, C.buf(), C.lead());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    if (M1::on_device) {
      synchronize();
    }
    cblas_syrk<T4>::func(CblasColMajor, cblas_uplo(uplo), cblas_trans(trans),
        C.size1(), k, alpha, A1->buf(), A1->lead(), beta, C.buf(), C.lead());
    delete A1;
  }
}

template<class M1, class M2>
void bi::potrf(const M1 A, M2 L, char uplo) {
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  assert (uplo == 'U' || uplo == 'L');
  assert (A.size1() == L.size1());
  assert (A.size2() == L.size2());
  assert (L.size1() == L.size2());

  int info;
  typename M2::size_type N = A.size1();
  typename M2::size_type ld = L.lead();

  L = A; /// @todo Single argument version to avoid unnecessary assignment
  if (M2::on_device) {
    magma_potrf<T2>::func(uplo, N, L.buf(), ld, &info);
    synchronize();
  } else {
    if (!M2::on_device && M1::on_device) {
      synchronize();
    }
    info = clapack_potrf<T2>::func(CblasColMajor, cblas_uplo(uplo), N,
        L.buf(), ld);
  }
  if (info != 0) { // try adjusting main diagonal
    BI_WARN(info == 0, "Cholesky failed with info " << info <<
        ", adjusting diagonal");

    BOOST_AUTO(d, diagonal(L));
    BOOST_AUTO(eps, temp_vector<M2>(d.size()));
    bi::fill(eps->begin(), eps->end(), static_cast<T2>(1.0));

    real smallest = *amin(d.begin(), d.end());
    real largest = *amax(d.begin(), d.end());
    real factor = pow(2.0, -std::numeric_limits<real>::digits);
    if (smallest > 0.0) {
      factor *= smallest;
    }

    while (info != 0 && factor < largest) {
      L = A;
      axpy(factor, *eps, d);
      if (M2::on_device) {
        magma_potrf<T2>::func(uplo, N, L.buf(), ld, &info);
        synchronize();
      } else {
        info = clapack_potrf<T2>::func(CblasColMajor, cblas_uplo(uplo), N,
            L.buf(), ld);
      }
      factor *= 2.0;
    }
    delete eps;
  }
  BI_ERROR(info == 0, "Cholesky failed with info " << info <<
      ", could not fix by adjusting diagonal");
}

template<class M1, class M2>
void bi::potrs(const M1 L, M2 X, char uplo) {
  typedef typename M1::value_type T1;
  typedef typename M2::value_type T2;

  /* pre-conditions */
  assert (uplo == 'U' || uplo == 'L');
  assert (L.size2() == X.size1());
  assert ((equals<T1,T2>::value));

  int info;
  if (M2::on_device) {
    BOOST_AUTO(L1, gpu_map_matrix(L));
    magma_potrs<T2>::func(uplo, L1->size1(), X.size2(), L1->buf(),
        L1->lead(), X.buf(), X.lead(), &info);
    synchronize();
    delete L1;
  } else {
    BOOST_AUTO(L1, host_map_matrix(L));
    if (M1::on_device) {
      synchronize();
    }
    info = clapack_potrs<T2>::func(CblasColMajor, cblas_uplo(uplo),
        L1->size1(), X.size2(), L1->buf(), L1->lead(), X.buf(), X.lead());
    delete L1;
  }
  BI_ASSERT(info == 0, "Solve failed with info " << info);
}

template<class T1, class M1, class M2>
void bi::trsm(const T1 alpha, const M1 A, M2 B, const char side,
    const char uplo = 'L', const char trans = 'N', const char diag = 'N') {
  typedef typename M1::value_type T2;
  typedef typename M2::value_type T3;

  /* pre-conditions */
  assert (side == 'L' || side == 'R');
  assert (uplo == 'U' || uplo == 'L');
  assert (trans == 'N' || trans == 'T');
  assert (diag == 'U' || diag == 'N');
  assert (!(trans == 'T' && side == 'L')  || A.size1() == B.size1());
  assert (!(trans == 'N' && side == 'L')  || A.size2() == B.size1());
  assert (!(trans == 'T' && side == 'R')  || B.size2() == A.size2());
  assert (!(trans == 'N' && side == 'R')  || B.size2() == A.size1());
  assert ((equals<T2,T3>::value));

  if (M2::on_device) {
    BOOST_AUTO(A1, gpu_map_matrix(A));
    cublas_trsm<T2>::func(side, uplo, trans, diag, B.size1(), B.size2(),
        alpha, A1->buf(), A1->lead(), B.buf(), B.lead());
    synchronize();
    CUBLAS_CHECK;
    delete A1;
  } else {
    BOOST_AUTO(A1, host_map_matrix(A));
    if (M1::on_device) {
      synchronize();
    }
    cblas_trsm<T2>::func(CblasColMajor, cblas_side(side),
        cblas_uplo(uplo), cblas_trans(trans), cblas_diag(diag), B.size1(),
        B.size2(), alpha, A1->buf(), A1->lead(), B.buf(), B.lead());
    delete A1;
  }
}

#endif
