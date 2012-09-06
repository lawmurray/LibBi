/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_EXPGAUSSIANPDF_HPP
#define BI_PDF_EXPGAUSSIANPDF_HPP

#include "GaussianPdf.hpp"
#include "LogTransformPdf.hpp"

#include "boost/serialization/split_member.hpp"

#include <set>

namespace bi {
/**
 * (log-)Gaussian distribution.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 *
 * This class encapsulates a Gaussian distribution, but where zero or more
 * variables may in fact be the logarithm of the true variables of interest.
 * Essentially it allows the combination of normal and log-normal variates
 * into one distribution, internally handling the @p exp() of log-variables
 * when sampling, and the @p log() of such variables during density
 * evaluations.
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class V1 = host_vector<real>, class M1 = host_matrix<real> >
class ExpGaussianPdf : public LogTransformPdf<GaussianPdf<V1,M1> > {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  ExpGaussianPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  ExpGaussianPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param ids Indices of log-variables.
   */
  ExpGaussianPdf(const int N, const std::set<int>& ids);

  /**
   * Construct univariate pdf.
   *
   * @param mu \f$\mu\f$; mean.
   * @param sigma \f$\sigma\f$; standard deviation.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  ExpGaussianPdf(const real mu, const real sigma, const bool log = false);

  /**
   * Construct univariate, zero-mean pdf.
   *
   * @param sigma2 \f$\sigma\f$; standard deviation.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  ExpGaussianPdf(const real sigma, const bool log = false);

  /**
   * Constructor.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   * @param ids Indices of log-variables.
   */
  template<class V2, class M2>
  ExpGaussianPdf(const V2 mu, const M2 U, const std::set<int>& ids);

  /**
   * Constructor.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   *
   * The pdf is initialised with no log-variables.
   */
  template<class V2, class M2>
  ExpGaussianPdf(const V2 mu, const M2 U);

  /**
   * Construct zero-mean pdf.
   *
   * @tparam M2 Matrix type.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   * @param ids Indices of log-variables.
   */
  template<class M2>
  ExpGaussianPdf(const M2 U, const std::set<int>& ids);

  /**
   * Construct zero-mean pdf.
   *
   * @tparam M2 Matrix type.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   *
   * The pdf is initialised with no log-variables.
   */
  template<class M2>
  ExpGaussianPdf(const M2 U);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class V2, class M2>
  ExpGaussianPdf<V1,M1>& operator=(const ExpGaussianPdf<V2,M2>& o);

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

#include "../misc/assert.hpp"

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf() {
  //
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const int N) :
    LogTransformPdf<GaussianPdf<V1,M1> >(N) {
  //
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const int N,
    const std::set<int>& ids) : LogTransformPdf<GaussianPdf<V1,M1> >(N, ids) {
  //
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const real mu,
    const real sigma, const bool log) :
    LogTransformPdf<GaussianPdf<V1,M1> >(1, log) {
  this->setMean(mu);
  this->setStd(sigma);
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const real sigma,
    const bool log) : LogTransformPdf<GaussianPdf<V1,M1> >(1, log) {
  this->setStd(sigma);
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const V2 mu, const M2 U,
    const std::set<int>& ids) :
    LogTransformPdf<GaussianPdf<V1,M1> >(mu.size(), ids) {
  this->setMean(mu);
  this->setStd(U);
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const V2 mu, const M2 U) :
    LogTransformPdf<GaussianPdf<V1,M1> >(mu.size()) {
  this->setMean(mu);
  this->setStd(U);
}

template<class V1, class M1>
template<class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const M2 U,
    const std::set<int>& ids) :
    LogTransformPdf<GaussianPdf<V1,M1> >(U.size1(), ids) {
  this->setStd(U);
}

template<class V1, class M1>
template<class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const M2 U) :
    LogTransformPdf<GaussianPdf<V1,M1> >(U.size1()) {
  this->setStd(U);
}

template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianPdf<V1,M1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<LogTransformPdf<GaussianPdf<V1,M1> > >(*this);
}

template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianPdf<V1,M1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<LogTransformPdf<GaussianPdf<V1,M1> > >(*this);
}

#endif
