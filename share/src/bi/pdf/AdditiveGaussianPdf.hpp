/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_ADDITIVEGAUSSIANPDF_HPP
#define BI_PDF_ADDITIVEGAUSSIANPDF_HPP

#include "GaussianPdf.hpp"

#include "boost/serialization/split_member.hpp"

#include <set>

namespace bi {
/**
 * Additive Gaussian conditional distribution.
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
class AdditiveGaussianPdf : public GaussianPdf<V1,M1> {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  AdditiveGaussianPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  AdditiveGaussianPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param logs Indices of log-variables.
   */
  AdditiveGaussianPdf(const int N, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   * @param logs Indices of log-variables.
   */
  AdditiveGaussianPdf(const M1& U, const std::set<int>& logs);

  /**
   * Construct multivariate, zero-mean pdf.
   *
   * @param U \f$U\f$; upper-triangular Cholesky factor of the covariance
   * matrix.
   *
   * The pdf is initialised with no log-variables.
   */
  AdditiveGaussianPdf(const M1& U);

  /**
   * Construct univariate, zero-mean pdf.
   *
   * @param sigma \f$\sigma\f$; standard deviation of the Gaussian.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  AdditiveGaussianPdf(const real sigma, const bool log = false);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class M2>
  AdditiveGaussianPdf<V1,M1>& operator=(const AdditiveGaussianPdf<M2>& o);

  /**
   * @copydoc concept::ConditionalPdf::sample()
   */
  template<class V2, class V3>
  void sample(Random& rng, const V2 x1, V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::samples()
   */
  template<class V2, class M2>
  void samples(Random& rng, const V2 x1, M2 X2);

  /**
   * @copydoc concept::ConditionalPdf::density()
   */
  template<class V2, class V3>
  real density(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::densities()
   */
  template<class V2, class M2, class V3>
  void densities(const V2 x1, const M2 X2, V3 p, const bool clear = false);

  /**
   * @copydoc concept::ConditionalPdf::logDensity()
   */
  template<class V2, class V3>
  real logDensity(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::logDensities()
   */
  template<class V2, class M2, class V3>
  void logDensities(const V2 x1, const M2 X2, V3 p, const bool clear = false);

  /**
   * @copydoc concept::ConditionalPdf::operator()()
   */
  template<class V2, class V3>
  real operator()(const V2 x1, const V3 x2);

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

#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../misc/assert.hpp"

#include <algorithm>

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf() {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(const int N)
    : GaussianPdf<V1,M1>(N) {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(const int N,
    const std::set<int>& logs) : GaussianPdf<V1,M1>(N, logs) {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(const M1& U,
    const std::set<int>& logs) : GaussianPdf<V1,M1>(U, logs) {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(const M1& U) :
    GaussianPdf<V1,M1>(U) {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(const real sigma,
    const bool log) :
    GaussianPdf<V1,M1>(sigma, log) {
  //
}

template<class V1, class M1>
template<class M2>
bi::AdditiveGaussianPdf<V1,M1>& bi::AdditiveGaussianPdf<V1,M1>::operator=(
    const AdditiveGaussianPdf<M2>& o) {
  ExpGaussianPdf<V1,M1>::operator=(o);

  return *this;
}

template<class V1, class M1>
template<class V2, class V3>
void bi::AdditiveGaussianPdf<V1,M1>::sample(Random& rng, const V2 x1,
    V3 x2) {
  GaussianPdf<V1,M1>::sample(rng, x2);
  axpy(1.0, x1, x2);
}

template<class V1, class M1>
template<class V2, class M2>
void bi::AdditiveGaussianPdf<V1,M1>::samples(Random& rng, const V2 x1,
    M2 X2) {
  GaussianPdf<V1,M1>::samples(rng, X2);
  add_rows(X2, x1);
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveGaussianPdf<V1,M1>::density(const V2 x1, const V3 x2) {
  typename sim_temp_vector<V3>::type z2(x2.size());
  sub_elements(x2, x1, z2);
  real p = GaussianPdf<V1,M1>::density(z2);
  if (!bi::is_finite(p)) {
    p = 0.0;
  }

  /* post-condition */
  BI_ASSERT(p >= 0.0);

  return p;
}

template<class V1, class M1>
template<class V2, class M2, class V3 p>
real bi::AdditiveGaussianPdf<V1,M1>::densities(const V2 x1, const M2 X2,
    V3 p, const bool clear) {
  typename sim_temp_matrix<M2>::type Z2(X2.size1(), X2.size2()));
  Z2 = X2;
  sub_rows(Z2, x1);
  GaussianPdf<V1,M1>::densities(Z2, p, clear);
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveGaussianPdf<V1,M1>::logDensity(const V2 x1, const V3 x2) {
  typename temp_vector<V3>::type z2(x2.size());
  sub_elements(x2, x1, z2);
  return GaussianPdf<V1,M1>::logDensity(z2);
}

template<class V1, class M1>
template<class V2, class M2, class V3>
real bi::AdditiveGaussianPdf<V1,M1>::logDensities(const V2 x1,
    const M2 X2, V3 p, const bool clear) {
  typename temp_matrix<M2>::type Z2(X2.size1(), X2.size2());
  Z2 = X2;
  sub_rows(Z2, x1);
  GaussianPdf<V1,M1>::logDensities(Z2, p, clear);
}

template<class V1, class M1>
template<class V2, class V3>
real bi::AdditiveGaussianPdf<V1,M1>::operator()(const V2 x1, const V3 x2) {
  return density(x1, x2);
}

template<class V1, class M1>
template<class Archive>
void bi::AdditiveGaussianPdf<V1,M1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object<ExpGaussianPdf<V1,M1> >(*this);
}

template<class V1, class M1>
template<class Archive>
void bi::AdditiveGaussianPdf<V1,M1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<ExpGaussianPdf<V1,M1> >(*this);
}

#endif
