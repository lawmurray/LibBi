/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_SAMPLE_HPP
#define BI_PDF_SAMPLE_HPP

#include "../random/Random.hpp"

namespace bi {
/**
 * Normalise log-weights.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] lws Log-weights.
 *
 * Renormalises log-weights for numerical range.
 */
template<class V1>
void renormalise(V1 lws);

/**
 * Histogram weighted sample set.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam V3 Vector type.
 * @tparam V4 Vector type.
 *
 * @param x Samples.
 * @param w Weights.
 * @param[out] c Histogram bin centres.
 * @param[out] h Histogram bin heights.
 *
 * Note that the length of @p c and @p h (which should be the same) implies
 * the number of bins in which to gather.
 */
template<class V1, class V2, class V3, class V4>
void hist(const V1 x, const V2 w, V3 c, V4 h);

/**
 * Compute unweighted mean of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param[out] mu Mean.
 */
template<class M1, class V1>
void mean(const M1 X, V1 mu);

/**
 * Compute weighted mean of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param w Weight vector. Need not be normalised.
 * @param[out] mu Mean.
 */
template<class M1, class V1, class V2>
void mean(const M1 X, const V1 w, V2 mu);

/**
 * Compute unweighted covariance of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam M2 Matrix type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param mu Sample mean.
 * @param[out] Sigma Covariance.
 *
 * @note Normalises by \f$N - 1\f$ for \f$N\f$ samples.
 */
template<class M1, class V1, class M2>
void cov(const M1 X, const V1 mu, M2 Sigma);

/**
 * Compute weighted covariance of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam M2 Matrix type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param w Weight vector. All weights must be positive, but need not be
 * normalised.
 * @param mu Sample mean.
 * @param[out] Sigma Covariance.
 *
 * @todo Document @p Wt.
 */
template<class M1, class V1, class V2, class M2>
void cov(const M1 X, const V1 w, const V2 mu, M2 Sigma);

/**
 * Compute unweighted variance of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param mu Sample mean.
 * @param[out] sigma Variance.
 *
 * @note Normalises by \f$N - 1\f$ for \f$N\f$ samples.
 */
template<class M1, class V1, class V2>
void var(const M1 X, const V1 mu, V2 sigma);

/**
 * Compute weighted variance of sample set.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam V3 Vector type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param w Weight vector. All weights must be positive, but need not be
 * normalised.
 * @param mu Sample mean.
 * @param[out] sigma Variance.
 *
 * @todo Document @p Wt.
 */
template<class M1, class V1, class V2, class V3>
void var(const M1 X, const V1 w, const V2 mu, V3 sigma);

/**
 * Compute unweighted cross-covariance of two sample sets.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type
 * @tparam M2 Matrix type
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam M3 Matrix type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param Y Sample set. Rows index samples, columns index variables.
 * @param muX Sample mean of @p X.
 * @param muY Sample mean of @p Y.
 * @param[out] SigmaXY Cross-covariance. Rows index variables in @p X,
 * columns variables in @p Y.
 *
 * @note Normalises by \f$N - 1\f$ for \f$N\f$ samples.
 */
template<class M1, class M2, class V1, class V2, class M3>
void cross(const M1 X, const M2 Y, const V1 muX, const V2 muY, M3 SigmaXY);

/**
 * Compute weighted cross-covariance of two sample sets.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type
 * @tparam M2 Matrix type
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam V3 Vector type.
 * @tparam M3 Matrix type.
 *
 * @param X Sample set. Rows index samples, columns index variables.
 * @param Y Sample set. Rows index samples, columns index variables.
 * @param w Weight vector. Need not be normalised.
 * @param muX Sample mean of @p X.
 * @param muY Sample mean of @p Y.
 * @param[out] SigmaXY Cross-covariance. Rows index variables in @p X,
 * columns variables in @p Y.
 *
 * @todo Document @p Wt.
 */
template<class M1, class M2, class V1, class V2, class V3, class M3>
void cross(const M1 X, const M2 Y, const V1 w, const V2 muX, const V3 muY,
    M3 SigmaXY);

/**
 * Exponentiate components of a vector.
 *
 * @ingroup math_pdf
 *
 * @tparam V2 Vector type.
 * @tparam V3 Integral vector type.
 *
 * @param[in,out] x Vector.
 * @param is Indices of variables to exponentiate.
 */
template<class V2, class V3>
void exp_vector(V2 x, const V3& is);

/**
 * Exponentiate columns of a matrix.
 *
 * @ingroup math_pdf
 *
 * @tparam M2 Matrix type.
 * @tparam V2 Integral vector type.
 *
 * @param[in,out] X Matrix.
 * @param is Indices of columns to exponentiate.
 */
template<class M2, class V2>
void exp_columns(M2 X, const V2& is);

/**
 * Exponentiate rows of a matrix.
 *
 * @ingroup math_pdf
 *
 * @tparam M2 Matrix type.
 * @tparam V2 Integral vector type.
 *
 * @param[in,out] X Matrix.
 * @param is Indices of rows to exponentiate.
 */
template<class M2, class V2>
void exp_rows(M2 X, const V2& is);

/**
 * Log components of a vector.
 *
 * @ingroup math_pdf
 *
 * @tparam V2 Vector type.
 * @tparam V3 Integral vector type.
 *
 * @param[in,out] x Vector.
 * @param is Indices of variables to log.
 */
template<class V2, class V3>
void log_vector(V2 x, const V3& is);

/**
 * Log columns of a matrix.
 *
 * @ingroup math_pdf
 *
 * @tparam M2 Matrix type.
 * @tparam V2 Integral vector type.
 *
 * @param[in,out] X Matrix.
 * @param is Indices of columns to log.
 */
template<class M2, class V2>
void log_columns(M2 X, const V2& is);

/**
 * Log rows of a matrix.
 *
 * @ingroup math_pdf
 *
 * @tparam M2 Matrix type.
 * @tparam V2 Integral vector type.
 *
 * @param[in,out] X Matrix.
 * @param is Indices of rows to log.
 */
template<class M2, class V2>
void log_rows(M2 X, const V2& is);

/**
 * Calculate determinant of Jacobian for transformation of log variables.
 *
 * @ingroup math_pdf
 *
 * @tparam V2 Vector type.
 * @tparam V3 Integral vector type.
 *
 * @param[in,out] x Vector.
 * @param is Indices of log-variables.
 */
template<class V2, class V3>
real det_vector(const V2 x, const V3& is);

/**
 * Calculate determinant of Jacobian for transformation of log variables, for
 * each row of a matrix.
 *
 * @ingroup math_pdf
 *
 * @tparam M2 Matrix type.
 * @tparam V2 Integral vector type.
 * @tparam V3 Vector type.
 *
 * @param X Matrix.
 * @param is Indices of rows to log.
 * @param[out] det Determinants.
 */
template<class M2, class V2, class V3>
void det_rows(const M2 X, const V2& is, V3 det);

}

#include "../math/misc.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

template<class V1>
void bi::renormalise(V1 lws) {
  thrust::replace_if(lws.begin(), lws.end(), is_not_finite_functor<real>(),
      bi::log(0.0));
  real mx = max_reduce(lws);
  if (is_finite(mx)) {
    sub_elements(lws, mx, lws);
  }
}

template<class V1, class V2, class V3, class V4>
void bi::hist(const V1 x, const V2 w, V3 c, V4 h) {
  /* pre-condition */
  BI_ASSERT(x.size() == w.size());
  BI_ASSERT(c.size() == h.size());
  BI_ASSERT(!V3::on_device);
  BI_ASSERT(!V4::on_device);

  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;

  const int P = x.size();
  const int B = c.size();
  T1 mx, mn;
  int i, j;
  typename temp_host_vector<T1>::type xSorted(P);
  typename temp_host_vector<T2>::type wSorted(P);
  xSorted = x;
  wSorted = w;

  bi::sort_by_key(xSorted, wSorted);
  mn = xSorted[0];
  mx = xSorted[xSorted.size() - 1];

  /* compute bin right edges */
  for (j = 0; j < B; ++j) {
    c[j] = mn + (j + 1)*(mx - mn)/B;
  }

  /* compute bin heights */
  h.clear();
  for (i = 0, j = 0; i < P; ++i) {
    if (xSorted[i] >= c[j] && j < B - 1) {
      ++j;
    }
    h[j] += wSorted[i];
  }

  /* compute bin centres */
  for (j = B - 1; j > 0; --j) {
    c[j] = 0.5*(c[j - 1] + c[j]);
  }
  c[0] = 0.5*(mn + c[0]);
}

template<class M1, class V1>
void bi::mean(const M1 X, V1 mu) {
  /* pre-condition */
  BI_ASSERT(X.size2() == mu.size());

  const int N = X.size1();
  typename sim_temp_vector<V1>::type w(N);
  set_elements(w, 1.0);
  gemv(1.0/N, X, w, 0.0, mu, 'T');
}

template<class M1, class V1, class V2>
void bi::mean(const M1 X, const V1 w, V2 mu) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == mu.size());
  BI_ASSERT(X.size1() == w.size());

  typedef typename V1::value_type T;

  T Wt = sum_reduce(w);
  gemv(1.0/Wt, X, w, 0.0, mu, 'T');
}

template<class M1, class V1, class M2>
void bi::cov(const M1 X, const V1 mu, M2 Sigma) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == mu.size());
  BI_ASSERT(Sigma.size1() == mu.size() && Sigma.size2() == mu.size());

  const int N = X.size1();
  typename sim_temp_matrix<M2>::type Y(X.size1(), X.size2());
  Y = X;
  sub_rows(Y, mu);
  syrk(1.0/(N - 1.0), Y, 0.0, Sigma, 'U', 'T');
}

template<class M1, class V1, class V2, class M2>
void bi::cov(const M1 X, const V1 w, const V2 mu, M2 Sigma) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == mu.size());
  BI_ASSERT(X.size1() == w.size());
  BI_ASSERT(Sigma.size1() == mu.size() && Sigma.size2() == mu.size());

  typedef typename V1::value_type T;
  typename sim_temp_matrix<M2>::type Y(X.size1(), X.size2());
  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  typename sim_temp_vector<V2>::type v(w.size());

  T Wt = sum_reduce(w);
  Y = X;
  sub_rows(Y, mu);
  sqrt_elements(w, v);
  gdmm(1.0, v, Y, 0.0, Z);
  syrk(1.0/Wt, Z, 0.0, Sigma, 'U', 'T');
  // alternative weight: 1.0/(Wt - W2t/Wt)
}

template<class M1, class V1, class V2>
void bi::var(const M1 X, const V1 mu, V2 sigma) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == mu.size());
  BI_ASSERT(sigma.size() == mu.size());

  const int N = X.size1();
  typename sim_temp_matrix<M1>::type Z(X.size1(), X.size2());
  Z = X;
  sub_rows(Z, mu);
  dot_columns(Z, sigma);
  scal(1.0/(N - 1.0), sigma);
}

template<class M1, class V1, class V2, class V3>
void bi::var(const M1 X, const V1 w, const V2 mu, V3 sigma) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == mu.size());
  BI_ASSERT(X.size1() == w.size());
  BI_ASSERT(sigma.size() == mu.size());

  typedef typename V1::value_type T1;
  typename sim_temp_matrix<M1>::type Z(X.size1(), X.size2());
  typename sim_temp_matrix<M1>::type Y(X.size1(), X.size2());
  typename sim_temp_vector<V2>::type v(w.size());

  T1 Wt = sum_reduce(w);
  Z = X;
  sub_rows(Z, mu);
  sqrt_elements(w, v);
  gdmm(1.0, v, Z, 0.0, Y);
  dot_columns(Y, sigma);
  divscal_elements(sigma, Wt, sigma);
  // alternative weight: 1.0/(Wt - W2t/Wt)
}

template<class M1, class M2, class V1, class V2, class M3>
void bi::cross(const M1 X, const M2 Y, const V1 muX, const V2 muY,
    M3 SigmaXY) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == muX.size());
  BI_ASSERT(Y.size2() == muY.size());
  BI_ASSERT(X.size1() == Y.size1());
  BI_ASSERT(SigmaXY.size1() == muX.size() && SigmaXY.size2() == muY.size());

  const int N = X.size1();

  gemm(1.0/(N - 1.0), X, Y, 0.0, SigmaXY, 'T', 'N');
  ger(-N/(N - 1.0), muX, muY, SigmaXY);
}

template<class M1, class M2, class V1, class V2, class V3, class M3>
void bi::cross(const M1 X, const M2 Y, const V1 w, const V2 muX,
    const V3 muY, M3 SigmaXY) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == muX.size());
  BI_ASSERT(Y.size2() == muY.size());
  BI_ASSERT(X.size1() == Y.size1());
  BI_ASSERT(X.size1() == w.size());
  BI_ASSERT(Y.size1() == w.size());
  BI_ASSERT(SigmaXY.size1() == muX.size() && SigmaXY.size2() == muY.size());

  typedef typename V1::value_type T;
  typename sim_temp_matrix<M3>::type Z(X.size1(), X.size2());

  T Wt = sum_reduce(w);
  T Wt2 = std::pow(Wt, 2);
  T W2t = sumsq_reduce(w);

  gdmm(1.0, w, X, 0.0, Z);
  gemm(1.0/Wt, Z, Y, 0.0, SigmaXY, 'T', 'N');
  ger(-1.0, muX, muY, SigmaXY);
  matrix_scal(1.0/(1.0 - W2t/Wt2), SigmaXY);
}

template<class V2, class V3>
inline void bi::exp_vector(V2 x, const V3& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(elem, subrange(x, *iter, 1));
    exp_elements(elem, elem);
  }
}

template<class M2, class V2>
inline void bi::exp_columns(M2 X, const V2& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(col, column(X, *iter));
    exp_elements(col, col);
  }
}

template<class M2, class V2>
inline void bi::exp_rows(M2 X, const V2& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(row1, row(X, *iter));
    exp_elements(row1, row1);
  }
}

template<class V2, class V3>
inline void bi::log_vector(V2 x, const V3& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(elem, subrange(x, *iter, 1));
    log_elements(elem, elem);
  }
}

template<class M2, class V2>
inline void bi::log_columns(M2 X, const V2& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(col, column(X, *iter));
    log_elements(col, col);
  }
}

template<class M2, class V2>
inline void bi::log_rows(M2 X, const V2& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  for (; iter != end; ++iter) {
    BOOST_AUTO(row1, row(X, *iter));
    log_elements(row1, row1);
  }
}

template<class V2, class V3>
real bi::det_vector(const V2 x, const V3& is) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());
  real det = 1.0;
  for (; iter != end; ++iter) {
    det *= *(x.begin() + *iter);
  }
  return det;
}

template<class M2, class V2, class V3>
void bi::det_rows(const M2 X, const V2& is, V3 det) {
  BOOST_AUTO(iter, is.begin());
  BOOST_AUTO(end, is.end());

  set_elements(det, 1.0);
  for (; iter != end; ++iter) {
    mul_elements(det, column(X, *iter), det);
  }
}

#endif
