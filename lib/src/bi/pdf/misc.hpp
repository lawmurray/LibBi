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
 * Rejection sample.
 *
 * @ingroup math_pdf
 *
 * @tparam Q1 Pdf type.
 * @tparam Q2 Pdf type.
 * @tparam V1 Vector type.
 *
 * @param rng Random number generator.
 * @param p Target distribution.
 * @param q Proposal distribution.
 * @param M Constant, such that \f$\forall \mathbf{x}: M q(\mathbf{x}) \ge
 * p(\mathbf{x})\f$.
 * @param[out] x \f$\mathbf{x} \sim p(\mathbf{x})\f$.
 */
template<class Q1, class Q2, class V1>
void rejectionSample(Random& rng, Q1& p, Q2& q, const real M, V1 x);

/**
 * Standardise samples.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param p Distribution providing mean and covariance with which to
 * standardise.
 * @param[in,out] X Samples. Rows index samples, columns variables.
 *
 * Standardises the samples @p X using the mean and covariance of the provide
 * distribution @p p. If this is the sample mean and covariance of the
 * samples, the output sample set will have mean zero and identity covariance.
 */
template<class V1, class M1, class M2>
void standardise(const ExpGaussianPdf<V1,M1>& p, M2 X);

/**
 * Condition (log-)Gaussian distribution.
 *
 * @ingroup math_pdf
 *
 * Considers a (log-)Gaussian distribution over a partitioned set
 * of variables \f$\{X_1,X_2\}\f$, \f$|X_1| = M\f$, \f$|X_2| = N\f$. For a
 * sample \f$\mathbf{x}_2\f$, computes \f$p(X_1|\mathbf{x}_2)\f$.
 *
 * @param p1 \f$p(X_1)\f$; marginal of variables in first partition.
 * @param p2 \f$p(X_2)\f$; marginal of variables in second partition.
 * @param C Cross-covariance matrix between \f$X_1\f$ and \f$X_2\f$, %size
 * \f$M \times N\f$.
 * @param x2 \f$\mathbf{x}_2\f$.
 * @param[out] p3 \f$p(X_1|\mathbf{x}_2)\f$.
 */
template<class V1, class M1, class V2, class M2, class M3, class V3,
    class V4, class M4>
void condition(const ExpGaussianPdf<V1,M1>& p1,
    const ExpGaussianPdf<V2,M2>& p2, const M3 C, const V3 x2,
    ExpGaussianPdf<V4,M4>& p3);

/**
 * Marginalise (log-)Gaussian distribution.
 *
 * @ingroup math_pdf
 *
 * Considers a (log-)Gaussian distribution over a partitioned set
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
template<class V1, class M1, class V2, class M2, class M3, class V4,
    class M4, class V5, class M5>
void marginalise(const ExpGaussianPdf<V1,M1>& p1,
    const ExpGaussianPdf<V2,M2>& p2, const M3 C,
    const ExpGaussianPdf<V4,M4>& q2, ExpGaussianPdf<V5,M5>& p3);

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
 * Compute distance matrix between samples.
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param X Samples. Rows index samples, columns variables.
 * @param h Gaussian kernel bandwidth.
 * @param[out] D Distance matrix (symmetric, upper diagonal stored).
 *
 * Computes the distance between all the samples of @p X, using a Euclidean
 * norm and Gaussian kernel of bandwidth @p h, writing results to @p D.
 */
template<class M1, class M2>
void distance(const M1 X, const real h, M2 D);

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
void cross(const M1 X, const M2 Y, const V1 muX, const V2 muY,
    M3 SigmaXY);

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
void cross(const M1 X, const M2 Y, const V1 w, const V2 muX,
    const V3 muY, M3 SigmaXY);

/**
 * Calculate \f$h_{opt}\f$.
 *
 * @ingroup math_pdf
 *
 * @param N Number of dimensions.
 * @param P Number of samples.
 *
 * Note:
 *
 * \f[
 * h_{opt} = \left[\frac{4}{(N+2)P}\right]^{\frac{1}{N+4}}\,,
 * \f]
 *
 * this being the optimal bandwidth for a kernel density estimate
 * over \f$P\f$ samples from a standard \f$N\f$-dimensional Gaussian
 * distribution, and Gaussian kernel (@ref Silverman1986
 * "Silverman, 1986"). We find this useful as a rule of thumb for
 * setting kernel density estimate bandwidths.
 *
 * @internal
 *
 * Imported from dysii 1.4.0, originally in indii/ml/aux/kde.hpp
 */
real hopt(const int N, const int P);

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
void expVec(V2 x, const V3& is);

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
void expCols(M2 X, const V2& is);

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
void expRows(M2 X, const V2& is);

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
void logVec(V2 x, const V3& is);

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
void logCols(M2 X, const V2& is);

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
void logRows(M2 X, const V2& is);

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
real detVec(const V2 x, const V3& is);

}

#include "AlmostGaussianKernel.hpp"

template<class Q1, class Q2, class V1>
inline void bi::rejectionSample(Random& rng, Q1& p, Q2& q, const real M,
    V1 x) {
  do {
    q.sample(rng, x);
  } while (rng.uniform<real>() > p(x)/(M*q(x)));
}

template<class V1, class M1, class M2>
void bi::standardise(const ExpGaussianPdf<V1,M1>& p, M2 X) {
  /* pre-condition */
  assert (p.size() == X.size2());

  logCols(X, p.getLogs());
  sub_rows(X, p.mean());
  trsm(1.0, p.std(), X, 'R', 'U');
  expCols(X, p.getLogs());
}

template<class V1, class M1, class V2, class M2, class M3, class V3,
    class V4, class M4>
void bi::condition(const ExpGaussianPdf<V1,M1>& p1,
    const ExpGaussianPdf<V2,M2>& p2, const M3 C, const V3 x2,
    ExpGaussianPdf<V4,M4>& p3) {
  /* pre-condition */
  assert (x2.size() == p2.size());
  assert (p3.size() == p1.size());
  assert (C.size1() == p1.size() && C.size2() == p2.size());

  BOOST_AUTO(z2, temp_vector<V1>(p2.size()));
  BOOST_AUTO(K, temp_matrix<M1>(p1.size(), p2.size()));

  /**
   * Compute gain matrix:
   *
   * \f[\mathcal{K} = C_{\mathbf{x}_1,\mathbf{x}_2}\Sigma_2^{-1}\,.\f]
   */
  symm(1.0, p2.prec(), C, 0.0, *K, 'R', 'U');

  /**
   * Then result is given by \f$\mathcal{N}(\boldsymbol{\mu}',
   * \Sigma')\f$, where:
   *
   * \f[\boldsymbol{\mu}' = \boldsymbol{\mu}_1 + \mathcal{K}(\mathbf{x}_2 -
   * \boldsymbol{\mu}_2)\,,\f]
   */
  *z2 = x2;
  logVec(*z2, p2.getLogs());
  axpy(-1.0, p2.mean(), *z2);
  p3.mean() = p1.mean();
  gemv(1.0, *K, *z2, 1.0, p3.mean());

  /**
   * and:
   *
   * \f{eqnarray*}
   * \Sigma' &=& \Sigma_1 - \mathcal{K}C_{\mathbf{x}_1,\mathbf{x}_2}^T \\
   * &=& \Sigma_1 - C_{\mathbf{x}_1,\mathbf{x}_2}\Sigma_2^{-1}C_{\mathbf{x}_1,\mathbf{x}_2}^T\,.
   * \f}
   */
  *K = C;
  trsm(1.0, p2.std(), *K, 'R', 'U');
  p3.cov() = p1.cov();
  syrk(-1.0, *K, 1.0, p3.cov(), 'U');

  /* update log-variables and precalculations */
  p3.setLogs(p1.getLogs());
  p3.init();

  /* clean up */
  synchronize();
  delete z2;
  delete K;
}

template<class V1, class M1, class V2, class M2, class M3, class V4,
    class M4, class V5, class M5>
void bi::marginalise(const ExpGaussianPdf<V1,M1>& p1,
    const ExpGaussianPdf<V2,M2>& p2, const M3 C,
    const ExpGaussianPdf<V4,M4>& q2, ExpGaussianPdf<V5,M5>& p3) {
  /* pre-conditions */
  assert (q2.size() == p2.size());
  assert (p3.size() == p1.size());
  assert (C.size1() == p1.size() && C.size2() == p2.size());

  BOOST_AUTO(z2, temp_vector<V1>(p2.size()));
  BOOST_AUTO(K, temp_matrix<M1>(p1.size(), p2.size()));
  BOOST_AUTO(A1, temp_matrix<M1>(p2.size(), p2.size()));
  BOOST_AUTO(A2, temp_matrix<M1>(p2.size(), p2.size()));

  /**
   * Compute gain matrix:
   *
   * \f[\mathcal{K} = C_{\mathbf{x}_1,\mathbf{x}_2}\Sigma_2^{-1}\,.\f]
   */
  symm(1.0, p2.prec(), C, 0.0, *K, 'R', 'U');

  /**
   * Then result is given by \f$\mathcal{N}(\boldsymbol{\mu}',
   * \Sigma')\f$, where:
   *
   * \f[\boldsymbol{\mu}' = \boldsymbol{\mu}_1 +
   * \mathcal{K}(\boldsymbol{\mu}_3 - \boldsymbol{\mu}_2)\,,\f]
   */
  *z2 = q2.mean();
  axpy(-1.0, p2.mean(), *z2);
  p3.mean() = p1.mean();
  gemv(1.0, *K, *z2, 1.0, p3.mean());

  /**
   * and:
   *
   * \f{eqnarray*}
   * \Sigma' &=& \Sigma_1 + \mathcal{K}(\Sigma_3 -
   * \Sigma_2)\mathcal{K}^T \\
   * &=& \Sigma_1 + \mathcal{K}\Sigma_3\mathcal{K}^T -
   * \mathcal{K}\Sigma_2\mathcal{K}^T\,.
   * \f}
   */
  p3.cov() = p1.cov();

  *A1 = *K;
  trmm(1.0, q2.std(), *A1, 'R', 'U', 'T');
  syrk(1.0, *A1, 1.0, p3.cov(), 'U');

  *A2 = *K;
  trmm(1.0, p2.std(), *A2, 'R', 'U', 'T');
  syrk(-1.0, *A2, 1.0, p3.cov(), 'U');

  /* make sure correct log-variables set */
  p3.setLogs(p2.getLogs());
  p3.init(); // redo precalculations

  /* clean up */
  delete z2;
  delete K;
  delete A1;
  delete A2;
}

template<class V1, class V2, class V3, class V4>
void bi::hist(const V1 x, const V2 w, V3 c, V4 h) {
  /* pre-condition */
  assert (x.size() == w.size());
  assert (c.size() == h.size());
  assert (!V3::on_device);
  assert (!V4::on_device);

  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;

  const int P = x.size();
  const int B = c.size();
  T1 mx, mn;
  int i, j;

  BOOST_AUTO(xSorted, host_temp_vector<T1>(P));
  BOOST_AUTO(wSorted, host_temp_vector<T2>(P));
  *xSorted = x;
  *wSorted = w;

  thrust::sort_by_key(xSorted->begin(), xSorted->end(), wSorted->begin());
  mn = (*xSorted)[0];
  mx = (*xSorted)[xSorted->size() - 1];

  /* compute bin right edges */
  for (j = 0; j < B; ++j) {
    c[j] = mn + (j + 1)*(mx - mn)/B;
  }

  /* compute bin heights */
  h.clear();
  for (i = 0, j = 0; i < P; ++i) {
    if ((*xSorted)[i] >= c[j] && j < B - 1) {
      ++j;
    }
    h[j] += (*wSorted)[i];
  }

  /* compute bin centres */
  for (j = B - 1; j > 0; --j) {
    c[j] = 0.5*(c[j - 1] + c[j]);
  }
  c[0] = 0.5*(mn + c[0]);

  delete xSorted;
  delete wSorted;
}

template<class M1, class M2>
void bi::distance(const M1 X, const real h, M2 D) {
  /* pre-conditions */
  assert (D.size1() == D.size2());
  assert (D.size1() == X.size1());
  assert (!M2::on_device);

  typedef typename M1::value_type T1;

  AlmostGaussianKernel K(X.size2(), h);
  BOOST_AUTO(d, host_temp_vector<T1>(X.size2()));
  int i, j;
  for (j = 0; j < D.size2(); ++j) {
    for (i = 0; i <= j; ++i) {
      *d = row(X,i);
      axpy(-1.0, row(X,j), *d);
      D(i,j) = K(dot(*d,*d));
    }
  }
  delete d;
}

template<class M1, class V1>
void bi::mean(const M1 X, V1 mu) {
  /* pre-condition */
  assert (X.size2() == mu.size());

  const int N = X.size1();

  BOOST_AUTO(w, temp_vector<V1>(N));
  bi::fill(w->begin(), w->end(), 1.0);
  gemv(1.0/N, X, *w, 0.0, mu, 'T');

  synchronize();
  delete w;
}

template<class M1, class V1, class V2>
void bi::mean(const M1 X, const V1 w, V2 mu) {
  /* pre-conditions */
  assert (X.size2() == mu.size());
  assert (X.size1() == w.size());

  typedef typename V1::value_type T;

  T Wt = sum(w.begin(), w.end(), static_cast<T>(0));

  gemv(1.0/Wt, X, w, 0.0, mu, 'T');
}

template<class M1, class V1, class M2>
void bi::cov(const M1 X, const V1 mu, M2 Sigma) {
  /* pre-conditions */
  assert (X.size2() == mu.size());
  assert (Sigma.size1() == mu.size() && Sigma.size2() == mu.size());

  const int N = X.size1();

  BOOST_AUTO(Y, temp_matrix<M2>(X.size1(), X.size2()));
  *Y = X;
  sub_rows(*Y, mu);
  syrk(1.0/(N - 1.0), *Y, 0.0, Sigma, 'U', 'T');

  synchronize();
  delete Y;
}

template<class M1, class V1, class V2, class M2>
void bi::cov(const M1 X, const V1 w, const V2 mu, M2 Sigma) {
  /* pre-conditions */
  assert (X.size2() == mu.size());
  assert (X.size1() == w.size());
  assert (Sigma.size1() == mu.size() && Sigma.size2() == mu.size());

  typedef typename V1::value_type T;

  BOOST_AUTO(Y, temp_matrix<M2>(X.size1(), X.size2()));
  BOOST_AUTO(Z, temp_matrix<M2>(X.size1(), X.size2()));
  BOOST_AUTO(v, temp_vector<V2>(w.size()));

  T Wt = sum(w.begin(), w.end(), static_cast<T>(0));
  //T W2t = sum_square(w.begin(), w.end(), static_cast<T>(0));

  *Y = X;
  sub_rows(*Y, mu);
  element_sqrt(w.begin(), w.end(), v->begin());
  gdmm(1.0, *v, *Y, 0.0, *Z);
  syrk(1.0/Wt, *Z, 0.0, Sigma, 'U', 'T');
  // alternative weight: 1.0/(Wt - W2t/Wt)

  synchronize();
  delete Y;
  delete Z;
  delete v;
}

template<class M1, class V1, class V2>
void bi::var(const M1 X, const V1 mu, V2 sigma) {
  /* pre-conditions */
  assert (X.size2() == mu.size());
  assert (sigma.size() == mu.size());

  const int N = X.size1();

  BOOST_AUTO(Z, temp_matrix<M1>(X.size2(), X.size1()));
  *Z = X;
  sub_rows(*Z, mu);
  dot_columns(*Z, sigma);
  scal(1.0/(N - 1.0), sigma);

  synchronize();
  delete Z;
}

template<class M1, class V1, class V2, class V3>
void bi::var(const M1 X, const V1 w, const V2 mu, V3 sigma) {
  /* pre-conditions */
  assert (X.size2() == mu.size());
  assert (X.size1() == w.size());
  assert (sigma.size() == mu.size());

  typedef typename V1::value_type T1;

  BOOST_AUTO(Z, temp_matrix<M1>(X.size1(), X.size2()));
  BOOST_AUTO(Y, temp_matrix<M1>(X.size1(), X.size2()));
  BOOST_AUTO(v, temp_vector<V2>(w.size()));

  T1 Wt = sum(w.begin(), w.end(), static_cast<T1>(0));
  //T1 W2t = sum_square(w.begin(), w.end(), static_cast<T1>(0));

  *Z = X;
  sub_rows(*Z, mu);
  element_sqrt(w.begin(), w.end(), v->begin());
  gdmm(1.0, *v, *Z, 0.0, *Y);
  dot_columns(*Y, sigma);
  scal(1.0/Wt, sigma);
  // alternative weight: 1.0/(Wt - W2t/Wt)

  synchronize();
  delete Y;
  delete Z;
  delete v;
}

template<class M1, class M2, class V1, class V2, class M3>
void bi::cross(const M1 X, const M2 Y, const V1 muX, const V2 muY,
    M3 SigmaXY) {
  /* pre-conditions */
  assert (X.size2() == muX.size());
  assert (Y.size2() == muY.size());
  assert (X.size1() == Y.size1());
  assert (SigmaXY.size1() == muX.size() && SigmaXY.size2() == muY.size());

  const int N = X.size1();

  gemm(1.0/(N - 1.0), X, Y, 0.0, SigmaXY, 'T', 'N');
  ger(-N/(N - 1.0), muX, muY, SigmaXY);
}

template<class M1, class M2, class V1, class V2, class V3, class M3>
void bi::cross(const M1 X, const M2 Y, const V1 w, const V2 muX,
    const V3 muY, M3 SigmaXY) {
  /* pre-conditions */
  assert (X.size2() == muX.size());
  assert (Y.size2() == muY.size());
  assert (X.size1() == Y.size1());
  assert (X.size1() == w.size());
  assert (Y.size1() == w.size());
  assert (SigmaXY.size1() == muX.size() && SigmaXY.size2() == muY.size());

  typedef typename V1::value_type T;

  BOOST_AUTO(Z, temp_matrix<M3>(X.size1(), X.size2()));

  T Wt = sum(w.begin(), w.end(), static_cast<T>(0));
  T Wt2 = std::pow(Wt, 2);
  T W2t = sum_square(w.begin(), w.end(), static_cast<T>(0));

  gdmm(1.0, w, X, 0.0, *Z);
  gemm(1.0/Wt, *Z, Y, 0.0, SigmaXY, 'T', 'N');
  ger(-1.0, muX, muY, SigmaXY);
  matrix_scal(1.0/(1.0 - W2t/Wt2), SigmaXY);

  synchronize();
  delete Z;
}

inline real bi::hopt(const int N, const int P) {
  return std::pow(4.0/((N+2)*P), 1.0/(N+4.0));
}

template<class V2, class V3>
inline void bi::expVec(V2 x, const V3& is) {
  typename V3::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(elem, subrange(x, *iter, 1));
    element_exp(elem.begin(), elem.end(), elem.begin());
  }
}

template<class M2, class V2>
inline void bi::expCols(M2 X, const V2& is) {
  typename V2::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(col, column(X, *iter));
    element_exp(col.begin(), col.end(), col.begin());
  }
}

template<class M2, class V2>
inline void bi::expRows(M2 X, const V2& is) {
  typename V2::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(row1, row(X, *iter));
    element_exp(row1.begin(), row1.end(), row1.begin());
  }
}

template<class V2, class V3>
inline void bi::logVec(V2 x, const V3& is) {
  typename V3::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(elem, subrange(x, *iter, 1));
    element_log(elem.begin(), elem.end(), elem.begin());
  }
}

template<class M2, class V2>
inline void bi::logCols(M2 X, const V2& is) {
  typename V2::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(col, column(X, *iter));
    element_log(col.begin(), col.end(), col.begin());
  }
}

template<class M2, class V2>
inline void bi::logRows(M2 X, const V2& is) {
  typename V2::const_iterator iter;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    BOOST_AUTO(row1, row(X, *iter));
    element_log(row1.begin(), row1.end(), row1.begin());
  }
}

template<class V2, class V3>
inline real bi::detVec(const V2 x, const V3& is) {
  typename V3::const_iterator iter;
  real det = 1.0;
  for (iter = is.begin(); iter != is.end(); ++iter) {
    det *= *(x.begin() + *iter);
  }
  return det;
}

#endif
