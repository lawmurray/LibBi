/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_GAUSSIANPDF_HPP
#define BI_PDF_GAUSSIANPDF_HPP

#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../random/Random.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

namespace bi {
/**
 * Multivariate Gaussian probability distribution.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 *
 * @section GaussianPdf_Serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class V1 = host_vector<real>, class M1 = host_matrix<real> >
class GaussianPdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the Gaussian with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  GaussianPdf();

  /**
   * Construct standard Gaussian (zero mean, identity covariance).
   *
   * @param N Number of dimensions.
   */
  GaussianPdf(const int N);

  /**
   * Construct univariate Gaussian.
   *
   * @param mu \f$\mu\f$; mean of the Gaussian.
   * @param sigma2 \f$\sigma^2\f$; variance of the Gaussian.
   */
  GaussianPdf(const real mu, const real sigma2);

  /**
   * Construct univariate, zero-mean Gaussian.
   *
   * @param sigma \f$\sigma\f$; variance of the Gaussian.
   */
  GaussianPdf(const real sigma2);

  /**
   * Construct multivariate Gaussian.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Vector type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean of the Gaussian.
   * @param Sigma \f$\Sigma\f$; covariance of the Gaussian.
   */
  template<class V2, class M2>
  GaussianPdf(const V2 mu, const M2 Sigma);

  /**
   * Construct multivariate, zero-mean Gaussian.
   *
   * @tparam M2 Vector type.
   *
   * @param sigma \f$\Sigma\f$; covariance of the Gaussian.
   */
  template<class M2>
  GaussianPdf(const M2 Sigma);

  /**
   * Copy constructor.
   */
  GaussianPdf(const GaussianPdf<V1,M1>& o);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class V2, class M2>
  GaussianPdf<V1,M1>& operator=(const GaussianPdf<V2,M2>& o);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * Resize.
   *
   * @param N Number of dimensions.
   * @param preserve True to preserve first @p N dimensions, false otherwise.
   */
  void resize(const int N, const bool preserve = true);

  /**
   * Get the mean.
   *
   * @return \f$\mathbf{\mu}\f$; mean.
   */
  V1& mean();

  /**
   * Get the covariance.
   *
   * @return \f$\Sigma\f$; covariance.
   */
  M1& cov();

  /**
   * Get the standard deviation (upper-triangular Cholesky factor of
   * covariance matrix).
   *
   * @return \f$\sqrt{\Sigma}\f$; standard deviation.
   *
   * Lower triangle of matrix is guaranteed zero.
   */
  M1& std();

  /**
   * Get the precision (inverse of covariance matrix).
   *
   * @return \f$\Sigma^{-1}\f$; precision matrix.
   */
  M1& prec();

  /**
   * @copydoc mean
   */
  const V1& mean() const;

  /**
   * @copydoc cov
   */
  const M1& cov() const;

  /**
   * @copydoc std
   */
  const M1& std() const;

  /**
   * @copydoc prec
   */
  const M1& prec() const;

  /**
   * Set the mean.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   */
  template<class V2>
  void setMean(const V2 mu);

  /**
   * Set the covariance.
   *
   * @param Sigma \f$\Sigma\f$; covariance.
   */
  template<class M2>
  void setCov(const M2 Sigma);

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class V2>
  void sample(Random& rng, V2 x);

  /**
   * @copydoc concept::Pdf::samples()
   */
  template<class M2>
  void samples(Random& rng, M2 X);

  /**
   * @copydoc concept::Pdf::density()
   */
  template<class V2>
  real density(const V2 x);

  /**
   * @copydoc concept::Pdf::densities()
   */
  template<class M2, class V2>
  void densities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  real logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::operator()(const V1)
   */
  template<class V2>
  real operator()(const V2 x);

  /**
   * Perform precalculations. This is called whenever setMean() or setCov()
   * are used, but should be called manually if get() is used to modify
   * mean and covariance directly.
   */
  void init();

protected:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * \f$\mu\f$; mean.
   */
  V1 mu;

  /**
   * \f$\Sigma\f$; covariance. Upper triangle stored.
   */
  M1 Sigma;

  /**
   * \f$U\f$ where \f$\Sigma = U^TU\f$ is the Cholesky factorisation
   * of the covariance matrix. Upper triangular.
   */
  M1 U;

  /**
   * \f$U^{-1}\f$. Upper-triangular.
   */
  M1 invU;

  /**
   * \f$\Sigma^{-1}\f$. Upper triangle stored.
   */
  M1 invSigma;

  /**
   * \f$\det(\Sigma)\f$
   */
  typename V1::value_type detSigma;

  /**
   * \f$\frac{1}{Z}\f$
   */
  typename V1::value_type invZ;

  /**
   * \f$\log Z\f$
   */
  typename V1::value_type logZ;

private:
  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
  #endif
};

}

#include "../math/pi.hpp"
#include "../math/view.hpp"
#include "../misc/assert.hpp"

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif
#include "boost/typeof/typeof.hpp"

template<class V1, class M1>
bi::GaussianPdf<V1,M1>::GaussianPdf() : N(0) {
  //
}

template<class V1, class M1>
bi::GaussianPdf<V1,M1>::GaussianPdf(const int N) :
    N(N), mu(N), Sigma(N,N), U(N,N), invU(N,N), invSigma(N,N) {
  mu.clear();
  ident(Sigma);
  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1,M1>::GaussianPdf(const real mu, const real sigma2) :
    N(1), mu(1), Sigma(1,1), U(1,1), invU(1), invSigma(1,1) {
  this->mu(0) = mu;
  this->Sigma(0,0) = sigma2;
  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1,M1>::GaussianPdf(const real sigma2) :
    N(1), mu(1), Sigma(1,1), U(1,1), invU(1,1), invSigma(1,1) {
  this->mu.clear();
  this->Sigma(0,0) = sigma2;
  init();
}

template<class V1, class M1>
template<class V2, class M2>
bi::GaussianPdf<V1,M1>::GaussianPdf(const V2 mu, const M2 Sigma) :
    N(mu.size()), mu(mu), Sigma(Sigma), U(N,N), invU(N,N), invSigma(N,N) {
  /* pre-condition */
  assert(mu.size() == Sigma.size1());
  assert(Sigma.size1() == Sigma.size2());

  init();
}

template<class V1, class M1>
template<class M2>
bi::GaussianPdf<V1,M1>::GaussianPdf(const M2 Sigma) :
    N(Sigma.size1()), mu(N), Sigma(Sigma), U(N,N), invU(N,N), invSigma(N,N) {
  mu.clear();
  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1,M1>::GaussianPdf(const GaussianPdf<V1,M1>& o) :
    N(o.N), mu(o.N), Sigma(o.N,o.N), U(o.N,o.N), invU(o.N,o.N),
    invSigma(o.N,o.N) {
  mu = o.mu;
  Sigma = o.Sigma;
  U = o.U;
  invSigma = o.invSigma;
  detSigma = o.detSigma;
  invZ = o.invZ;
  logZ = o.logZ;
}

template<class V1, class M1>
template<class V2, class M2>
bi::GaussianPdf<V1,M1>& bi::GaussianPdf<V1,M1>::operator=(
    const GaussianPdf<V2,M2>& o) {
  /* pre-condition */
  assert (o.N == N);

  mu = o.mu;
  Sigma = o.Sigma;
  U = o.U;
  invU = o.invU;
  invSigma = o.invSigma;
  detSigma = o.detSigma;
  invZ = o.invZ;
  logZ = o.logZ;

  return *this;
}

template<class V1, class M1>
inline int bi::GaussianPdf<V1,M1>::size() const {
  return N;
}

template<class V1, class M1>
void bi::GaussianPdf<V1,M1>::resize(const int N, const bool preserve) {
  this->N = N;
  mu.resize(N, preserve);
  Sigma.resize(N, N, preserve);
  U.resize(N, N);
  invU.resize(N, N);
  invSigma.resize(N, N);
  if (preserve) {
    init();
  }
}

template<class V1, class M1>
inline V1& bi::GaussianPdf<V1,M1>::mean() {
  return mu;
}

template<class V1, class M1>
inline M1& bi::GaussianPdf<V1,M1>::cov() {
  return Sigma;
}

template<class V1, class M1>
inline M1& bi::GaussianPdf<V1,M1>::std() {
  return U;
}

template<class V1, class M1>
inline M1& bi::GaussianPdf<V1,M1>::prec() {
  return invSigma;
}

template<class V1, class M1>
inline const V1& bi::GaussianPdf<V1,M1>::mean() const {
  return mu;
}

template<class V1, class M1>
inline const M1& bi::GaussianPdf<V1,M1>::cov() const {
  return Sigma;
}

template<class V1, class M1>
inline const M1& bi::GaussianPdf<V1,M1>::std() const {
  return U;
}

template<class V1, class M1>
inline const M1& bi::GaussianPdf<V1,M1>::prec() const {
  return invSigma;
}

template<class V1, class M1>
template<class V2>
inline void bi::GaussianPdf<V1,M1>::setMean(const V2 mu) {
  /* pre-condition */
  assert(mu.size() == N); // new same size as old

  this->mu = mu;
  //init(); // not necessary
}

template<class V1, class M1>
template<class M2>
inline void bi::GaussianPdf<V1,M1>::setCov(const M2 Sigma) {
  /* pre-condition */
  assert(Sigma.size1() == N); // new same size as old

  this->Sigma = Sigma;
  init();
}

template<class V1, class M1>
template<class V2>
inline void bi::GaussianPdf<V1,M1>::sample(Random& rng, V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  rng.gaussians(x);
  trmv(U, x, 'U', 'T');
  axpy(1.0, mu, x);
}

template<class V1, class M1>
template<class M2>
void bi::GaussianPdf<V1,M1>::samples(Random& rng, M2 X) {
  /* pre-conditions */
  assert (X.size2() == size());

  rng.gaussians(matrix_as_vector(X));
  trmm(1.0, U, X, 'R', 'U');
  add_rows(X, mu);
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1,M1>::density(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  real exponent, p;
  BOOST_AUTO(z, temp_vector<V2>(N));
  *z = x;
  axpy(-1.0, mu, *z);
  trmv(invU, *z, 'U');
  exponent = -0.5*dot(*z, *z);
  p = invZ*exp(exponent);
  if (isnan(p)) {
    p = 0.0;
  }

  /* post-condition */
  assert (p >= 0.0);

  synchronize();
  delete z;

  return p;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::GaussianPdf<V1,M1>::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

  ///@todo Try to combine some of these operations.
  BOOST_AUTO(Z, temp_matrix<M2>(X.size2(), X.size1()));
  trans(X, *Z);
  sub_columns(*Z, mu);
  trmm(1.0, invU, *Z, 'L', 'U', 'T');
  dot_columns(*Z, p);
  scal(-0.5, p);
  element_exp(p.begin(), p.end(), p.begin());
  scal(invZ, p);

  synchronize();
  delete Z;
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1,M1>::logDensity(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  real exponent, p;
  BOOST_AUTO(z, temp_vector<V2>(N));
  *z = x;
  axpy(-1.0, mu, *z);
  trmv(invU, *z, 'U', 'T');
  exponent = -0.5*dot(*z, *z);
  p = exponent - logZ;

  synchronize();
  delete z;

  return p;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::GaussianPdf<V1,M1>::logDensities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

  typedef typename V1::value_type T2;

  ///@todo Try to combine some of these operations.
  BOOST_AUTO(Z, temp_matrix<M2>(X.size2(), X.size1()));
  transpose(X, *Z);
  sub_columns(*Z, mu);
  trmm(1.0, invU, *Z, 'L', 'U', 'T');
  dot_columns(*Z, p);
  scal(-0.5, p);
  thrust::transform(p.begin(), p.end(), p.begin(),
      subtract_constant_functor<T2>(logZ));

  synchronize();
  delete Z;
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1,M1>::operator()(const V2 x) {
  return density(x);
}

template<class V1, class M1>
void bi::GaussianPdf<V1,M1>::init() {
  if (N > 0) {
    U.clear();
    invU.clear();
    invSigma.clear();

    /* Cholesky decomposition of covariance matrix */
    potrf(Sigma, U, 'U');
    ident(invU);
    trsm(1.0, U, invU, 'R', 'U');

    /* inverse of covariance matrix */
    invSigma = invU;
    trmm(1.0, invU, invSigma, 'R', 'U', 'T');

    /* Determinant of covariance matrix, exploiting previous Cholesky. For
     * any matrices \f$A\f$ and \f$B\f$, \f$|AB| = |A||B|\f$. Given that
     * \f$\Sigma = LL^T\f$, \f$|\Sigma| = |L|^2\f$. The determinant of a
     * triangular matrix is simply the product of the elements on its
     * diagonal, so \f$|L|\f$ is easy to calculate. */
    BOOST_AUTO(d, diagonal(U));
    BOOST_TYPEOF(detSigma) detL = bi::prod(d.begin(), d.end(), REAL(1.0));
    detSigma = detL*detL;

    /* normalising constant for Gaussian, \f$Z = 1/\sqrt{(2\pi)^N|\Sigma|}\f$ */
    logZ = N*BI_HALF_LOG_TWO_PI + log(detL);
    invZ = exp(-logZ);
  }
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::GaussianPdf<V1,M1>::save(Archive& ar, const unsigned version) const {
  ar & N;
  ar & mu;
  ar & Sigma;
}

template<class V1, class M1>
template<class Archive>
void bi::GaussianPdf<V1,M1>::load(Archive& ar, const unsigned version) {
  ar & N;

  mu.resize(N, false);
  Sigma.resize(N, N, false);
  U.resize(N, N, false);
  invU.resize(N, N, false);
  invSigma.resize(N, N, false);

  ar & mu;
  ar & Sigma;

  init();
}
#endif

#endif
