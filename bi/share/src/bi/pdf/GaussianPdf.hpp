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
#include "primitive.hpp"

#include "boost/serialization/split_member.hpp"

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
template<class V1 = host_vector<real> , class M1 = host_matrix<real> >
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
   * @param sigma \f$\sigma\f$; standard deviation of the Gaussian.
   */
  GaussianPdf(const real mu, const real sigma);

  /**
   * Construct univariate, zero-mean Gaussian.
   *
   * @param sigma \f$\sigma\f$; standard deviation of the Gaussian.
   */
  GaussianPdf(const real sigma);

  /**
   * Construct multivariate Gaussian.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Vector type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean of the Gaussian.
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   */
  template<class V2, class M2>
  GaussianPdf(const V2 mu, const M2 U);

  /**
   * Construct multivariate, zero-mean Gaussian.
   *
   * @tparam M2 Matrix type.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   */
  template<class M2>
  GaussianPdf(const M2 Sigma);

  /**
   * Copy constructor.
   */
  GaussianPdf(const GaussianPdf<V1, M1>& o);

  /**
   * Generic copy constructor.
   */
  template<class V2, class M2>
  GaussianPdf(const GaussianPdf<V2, M2>& o);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  GaussianPdf<V1, M1>& operator=(const GaussianPdf<V1, M1>& o);

  /**
   * Generic assignment operator. Both sides must have the same
   * dimensionality.
   */
  template<class V2, class M2>
  GaussianPdf<V1, M1>& operator=(const GaussianPdf<V2, M2>& o);

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
   * Get upper-triangular Cholesky factor of the covariance matrix.
   *
   * @return \f$U\f$; the upper-triangular Cholesky factor of the covariance
   * matrix.
   */
  M1& std();

  /**
   * @copydoc mean
   */
  const V1& mean() const;

  /**
   * @copydoc std
   */
  const M1& std() const;

  /**
   * Get the determinant of the Cholesky factor of the covariance matrix.
   *
   * @return \f$|U|\f$; determinant of Cholesky factor of the covariance
   * matrix.
   */
  typename V1::value_type det() const;

  /**
   * Set the mean.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   */
  template<class V2>
  void setMean(const V2 mu);

  /**
   * Set the Cholesky factor of the covariance matrix.
   *
   * @param U \f$U\f$; the Cholesky factor of the covariance matrix.
   */
  template<class M2>
  void setStd(const M2 U);

  /**
   * Set the mean for a univariate distribution.
   *
   * @param mu \f$\mu\f$; mean.
   */
  void setMean(const real mu);

  /**
   * Set the standard deviation for a univariate distribution.
   *
   * @param sigma \f$\sigma\f$; the standard deviation.
   */
  void setStd(const real U);

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
  void densities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  real logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::operator()(const V1)
   */
  template<class V2>
  real operator()(const V2 x);

  /**
   * Perform precalculations. This is called automatically whenever setMean()
   * and setStd() are used, but should be called manually whenever mean() or
   * std() is used to modify the mean and Cholesky factor directly.
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
   * \f$U\f$ where \f$\Sigma = U^TU\f$ is the Cholesky factorisation
   * of the covariance matrix. Upper triangular.
   */
  M1 U;

  /**
   * \f$\|U|\f$
   */
  typename V1::value_type detU;

  /**
   * \f$\frac{1}{Z}\f$
   */
  typename V1::value_type invZ;

  /**
   * \f$\log Z\f$
   */
  typename V1::value_type logZ;

private:
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
};

}

#include "functor.hpp"
#include "../math/pi.hpp"
#include "../math/view.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../misc/assert.hpp"

#include "boost/serialization/base_object.hpp"

template<class V1, class M1>
bi::GaussianPdf<V1, M1>::GaussianPdf() :
  N(0) {
  //
}

template<class V1, class M1>
bi::GaussianPdf<V1, M1>::GaussianPdf(const int N) :
  N(N), mu(N), U(N, N) {
  mu.clear();
  ident(U);
  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1, M1>::GaussianPdf(const real mu, const real sigma) :
  N(1), mu(1), U(1, 1) {
  this->mu(0) = mu;
  this->U(0, 0) = sigma;
  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1, M1>::GaussianPdf(const real sigma) :
  N(1), mu(1), U(1, 1) {
  this->mu.clear();
  this->U(0, 0) = sigma;
  init();
}

template<class V1, class M1>
template<class V2, class M2>
bi::GaussianPdf<V1, M1>::GaussianPdf(const V2 mu, const M2 U) :
  N(mu.size()), mu(N), U(N, N) {
  /* pre-condition */
  BI_ASSERT(U.size1() == N);
  BI_ASSERT(U.size1() == U.size2());

  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 or M1 == M2 */
  this->mu = mu;
  this->U = U;

  init();
}

template<class V1, class M1>
template<class M2>
bi::GaussianPdf<V1, M1>::GaussianPdf(const M2 U) :
  N(U.size1()), mu(N), U(N, N) {
  /* pre-condition */
  BI_ASSERT(U.size1() == U.size2());

  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 */
  this->mu.clear();
  this->U = U;

  init();
}

template<class V1, class M1>
bi::GaussianPdf<V1, M1>::GaussianPdf(const GaussianPdf<V1, M1>& o) :
  N(o.N), mu(o.N), U(o.N, o.N) {
  operator=(o);
}

template<class V1, class M1>
template<class V2, class M2>
bi::GaussianPdf<V1, M1>::GaussianPdf(const GaussianPdf<V2, M2>& o) :
  N(o.N), mu(o.N), U(o.N, o.N) {
  operator=(o);
}

template<class V1, class M1>
bi::GaussianPdf<V1, M1>& bi::GaussianPdf<V1, M1>::operator=(const GaussianPdf<
    V1, M1>& o) {
  /* pre-condition */
  BI_ASSERT(o.N == N);

  mu = o.mu;
  U = o.U;
  detU = o.detU;
  invZ = o.invZ;
  logZ = o.logZ;

  return *this;
}

template<class V1, class M1>
template<class V2, class M2>
bi::GaussianPdf<V1, M1>& bi::GaussianPdf<V1, M1>::operator=(const GaussianPdf<
    V2, M2>& o) {
  /* pre-condition */
  BI_ASSERT(o.N == N);

  mu = o.mu;
  U = o.U;
  detU = o.detU;
  invZ = o.invZ;
  logZ = o.logZ;

  return *this;
}

template<class V1, class M1>
inline int bi::GaussianPdf<V1, M1>::size() const {
  return N;
}

template<class V1, class M1>
void bi::GaussianPdf<V1, M1>::resize(const int N, const bool preserve) {
  this->N = N;
  mu.resize(N, preserve);
  U.resize(N, N, preserve);
  if (preserve) {
    init();
  }
}

template<class V1, class M1>
inline V1& bi::GaussianPdf<V1, M1>::mean() {
  return mu;
}

template<class V1, class M1>
inline M1& bi::GaussianPdf<V1, M1>::std() {
  return U;
}

template<class V1, class M1>
inline const V1& bi::GaussianPdf<V1, M1>::mean() const {
  return mu;
}

template<class V1, class M1>
inline const M1& bi::GaussianPdf<V1, M1>::std() const {
  return U;
}

template<class V1, class M1>
inline typename V1::value_type bi::GaussianPdf<V1, M1>::det() const {
  return detU;
}

template<class V1, class M1>
template<class V2>
inline void bi::GaussianPdf<V1, M1>::setMean(const V2 mu) {
  /* pre-condition */
  BI_ASSERT(mu.size() == N);

  this->mu = mu;
  //init(); // not necessary
}

template<class V1, class M1>
template<class M2>
inline void bi::GaussianPdf<V1, M1>::setStd(const M2 U) {
  /* pre-condition */
  BI_ASSERT(U.size1() == N);

  this->U = U;
  init();
}

template<class V1, class M1>
inline void bi::GaussianPdf<V1, M1>::setMean(const real mu) {
  this->mu(0) = mu;
  //init(); // not necessary
}

template<class V1, class M1>
inline void bi::GaussianPdf<V1, M1>::setStd(const real sigma) {
  this->U(0, 0) = sigma;
  init();
}

template<class V1, class M1>
template<class V2>
inline void bi::GaussianPdf<V1, M1>::sample(Random& rng, V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  rng.gaussians(x);
  trmv(U, x, 'U', 'T');
  axpy(1.0, mu, x);
}

template<class V1, class M1>
template<class M2>
void bi::GaussianPdf<V1, M1>::samples(Random& rng, M2 X) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == size());

  rng.gaussians(vec(X));
  trmm(1.0, U, X, 'R', 'U');
  add_rows(X, mu);
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1, M1>::density(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  typename sim_temp_vector<V2>::type z(N);
  sub_elements(x, mu, z);
  trsv(U, z, 'U');
  real p = invZ*exp(-0.5*dot(z));
  if (bi::isnan(p)) {
    p = 0.0;
  }

  /* post-condition */
  BI_ASSERT(p >= 0.0);

  return p;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::GaussianPdf<V1, M1>::densities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  sub_rows(Z, mu);
  trsm(1.0, U, Z, 'R', 'U');
  gaussian_densities(Z, logZ, p, clear);
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1, M1>::logDensity(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  typename sim_temp_vector<V2>::type z(N);
  sub_elements(x, mu, z);
  trsv(U, z, 'U');

  return -0.5*dot(z) - logZ;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::GaussianPdf<V1, M1>::logDensities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  sub_rows(Z, mu);
  trsm(1.0, U, Z, 'R', 'U');
  gaussian_log_densities(Z, logZ, p, clear);
}

template<class V1, class M1>
template<class V2>
real bi::GaussianPdf<V1, M1>::operator()(const V2 x) {
  return density(x);
}

template<class V1, class M1>
void bi::GaussianPdf<V1, M1>::init() {
  /* determinant of Cholesky factor (product of the elements on its main
   * diagonal) */
  detU = prod_reduce(diagonal(U));

  /* normalising constant for Gaussian, \f$Z = 1/\sqrt{(2\pi)^N|\Sigma|}\f$ */
  logZ = N*BI_HALF_LOG_TWO_PI + log(detU);
  invZ = exp(-logZ);
}

template<class V1, class M1>
template<class Archive>
void bi::GaussianPdf<V1, M1>::save(Archive& ar, const unsigned version) const {
  ar & N;
  ar & mu;
  ar & U;
}

template<class V1, class M1>
template<class Archive>
void bi::GaussianPdf<V1, M1>::load(Archive& ar, const unsigned version) {
  ar & N;

  mu.resize(N, false);
  U.resize(N, N, false);

  ar & mu;
  ar & U;

  init();
}

#endif
