/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_EXPADDITIVEGAUSSIANPDF_HPP
#define BI_PDF_EXPADDITIVEGAUSSIANPDF_HPP

#include "AdditiveGaussianPdf.hpp"
#include "LogTransformConditionalPdf.hpp"

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
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
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
class ExpAdditiveGaussianPdf : public LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> > {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  ExpAdditiveGaussianPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  ExpAdditiveGaussianPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param logs Indices of log-variables.
   */
  ExpAdditiveGaussianPdf(const int N, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   * @param logs Indices of log-variables.
   */
  ExpAdditiveGaussianPdf(const M1& U, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   *
   * The pdf is initialised with no log-variables.
   */
  ExpAdditiveGaussianPdf(const M1& U);

  /**
   * Construct univariate, zero-mean pdf.
   *
   * @param sigma \f$\sigma\f$; standard deviation of the Gaussian.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  ExpAdditiveGaussianPdf(const real sigma, const bool log = false);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class M2>
  ExpAdditiveGaussianPdf<V1,M1>& operator=(const ExpAdditiveGaussianPdf<M2>& o);

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
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf() {
  //
}

template<class V1, class M1>
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf(const int N) :
    LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >(N) {
  //
}

template<class V1, class M1>
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf(const int N,
    const std::set<int>& logs) :
    LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >(N, logs) {
  //
}

template<class V1, class M1>
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf(const M1& U,
    const std::set<int>& logs) :
    LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >(U, logs) {
  //
}

template<class V1, class M1>
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf(const M1& U) :
    LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >(U) {
  //
}

template<class V1, class M1>
bi::ExpAdditiveGaussianPdf<V1,M1>::ExpAdditiveGaussianPdf(
    const real sigma, const bool log) :
    LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >(sigma, log) {
  //
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpAdditiveGaussianPdf<V1,M1>& bi::ExpAdditiveGaussianPdf<V1,M1>::operator=(
    const ExpAdditiveGaussianPdf<V2,M2>& o) {
  LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> >::operator=(o);

  return *this;
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::ExpAdditiveGaussianPdf<V1,M1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object<LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> > >(*this);
}

template<class V1, class M1>
template<class Archive>
void bi::ExpAdditiveGaussianPdf<V1,M1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<LogTransformConditionalPdf<AdditiveGaussianPdf<V1,M1> > >(*this);
}
#endif

#endif
