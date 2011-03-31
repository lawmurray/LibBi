/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_ADDITIVEEXPGAUSSIANPDF_HPP
#define BI_PDF_ADDITIVEEXPGAUSSIANPDF_HPP

#include "ExpGaussianPdf.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

#include <set>

namespace bi {
/**
 * Additive Gaussian conditional distribution with @c exp transformation of
 * zero or more variables.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Type of covariance matrix. Must be symmetric_matrix,
 * identity_matrix or diagonal_matrix.
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::ConditionalPdf
 */
template<class V1 = host_vector<>, class M1 = host_matrix<> >
class AdditiveExpGaussianPdf : protected ExpGaussianPdf<V1,M1> {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  AdditiveExpGaussianPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  AdditiveExpGaussianPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param logs Indices of log-variables.
   */
  AdditiveExpGaussianPdf(const int N, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param Sigma \f$\Sigma\f$; covariance.
   * @param logs Indices of log-variables.
   */
  AdditiveExpGaussianPdf(const M1& Sigma, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param Sigma \f$\Sigma\f$; covariance.
   *
   * The pdf is initialised with no log-variables.
   */
  AdditiveExpGaussianPdf(const M1& Sigma);

  /**
   * Construct univariate, zero-mean pdf.
   *
   * @param sigma2 \f$\sigma^2\f$; variance of the Gaussian.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  AdditiveExpGaussianPdf(const real sigma2, const bool log = false);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class M2>
  AdditiveExpGaussianPdf<V1,M1>& operator=(const AdditiveExpGaussianPdf<M2>& o);

  using ExpGaussianPdf<V1,M1>::init;
  using ExpGaussianPdf<V1,M1>::size;
  using ExpGaussianPdf<V1,M1>::mean;
  using ExpGaussianPdf<V1,M1>::cov;
  using ExpGaussianPdf<V1,M1>::std;
  using ExpGaussianPdf<V1,M1>::prec;
  using ExpGaussianPdf<V1,M1>::setMean;
  using ExpGaussianPdf<V1,M1>::setCov;
  using ExpGaussianPdf<V1,M1>::getLogs;
  using ExpGaussianPdf<V1,M1>::setLogs;
  using ExpGaussianPdf<V1,M1>::addLog;
  using ExpGaussianPdf<V1,M1>::addLogs;

  /**
   * @copydoc concept::ConditionalPdf::sample()
   */
  template<class V2, class V3>
  void sample(Random& rng, const V2 x1, V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::density()
   */
  template<class V2, class V3>
  real density(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::logDensity()
   */
  template<class V2, class V3>
  real logDensity(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::operator()()
   */
  template<class V2, class V3>
  real operator()(const V2 x1, const V3 x2);

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

#include "../misc/assert.hpp"

#include <algorithm>

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf() {
  //
}

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf(const int N)
    : ExpGaussianPdf<V1,M1>(N) {
  //
}

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf(const int N,
    const std::set<int>& logs) : ExpGaussianPdf<V1,M1>(N, logs) {
  //
}

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf(const M1& Sigma,
    const std::set<int>& logs) : ExpGaussianPdf<V1,M1>(Sigma, logs) {
  //
}

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf(const M1& Sigma) :
    ExpGaussianPdf<V1,M1>(Sigma) {
  //
}

template<class V1, class M1>
bi::AdditiveExpGaussianPdf<V1,M1>::AdditiveExpGaussianPdf(
    const real sigma2, const bool log) :
    ExpGaussianPdf<V1,M1>(sigma2, log) {
  //
}

template<class V1, class M1>
template<class M2>
bi::AdditiveExpGaussianPdf<V1,M1>& bi::AdditiveExpGaussianPdf<V1,M1>::operator=(
    const AdditiveExpGaussianPdf<M2>& o) {
  ExpGaussianPdf<V1,M1>::operator=(o);

  return *this;
}

template<class V1, class M1>
template<class V2, class V3>
void bi::AdditiveExpGaussianPdf<V1,M1>::sample(Random& rng, const V2 x1,
    V3 x2) {
  BOOST_AUTO(z, temp_vector<V2>(x1.size()));
  *z = x1;
  logVec(*z, this->getLogs());
  GaussianPdf<V1,M1>::sample(rng, x2);
  axpy(1.0, *z, x2);
  expVec(x2, this->getLogs());

  delete z;
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveExpGaussianPdf<V1,M1>::density(const V2 x1,
    const V3 x2) {
  BOOST_AUTO(z1, temp_vector<V2>(x1.size()));
  BOOST_AUTO(z2, temp_vector<V3>(x2.size()));

  real detJ, p;
  detJ = detVec(x2, this->getLogs()); // determinant of Jacobian for change of variable, x = exp(z)

  *z1 = x1;
  logVec(*z1, this->getLogs());
  *z2 = x2;
  logVec(*z2, this->getLogs());
  axpy(-1.0, *z1, *z2);
  p = GaussianPdf<V1,M1>::operator()(*z2)/detJ;

  delete z1;
  delete z2;

  /* post-condition */
  if (!IS_FINITE(p)) {
    p = 0.0;
  }
  assert(p >= 0.0);

  return p;
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveExpGaussianPdf<V1,M1>::logDensity(const V2 x1,
    const V3 x2) {
  return log(density(x1, x2));
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveExpGaussianPdf<V1,M1>::operator()(const V2 x1,
    const V3 x2) {
  return density(x1, x2);
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::AdditiveExpGaussianPdf<V1,M1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object<ExpGaussianPdf<V1,M1> >(*this);
}

template<class V1, class M1>
template<class Archive>
void bi::AdditiveExpGaussianPdf<V1,M1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<ExpGaussianPdf<V1,M1> >(*this);
}
#endif

#endif
