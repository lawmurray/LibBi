/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_ADDITIVEGAUSSIANPDF_HPP
#define BI_PDF_ADDITIVEGAUSSIANPDF_HPP

#include "GaussianPdf.hpp"

#ifndef __CUDACC__
#include "boost/serialization/serialization.hpp"
#endif

namespace bi {
/**
 * Additive Gaussian conditional probability distribution.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Type of mean.
 * @tparam M1 Type of covariance.
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
   * Initialises the pdf with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  AdditiveGaussianPdf();

  /**
   * Construct univariate pdf.
   *
   * @param sigma \f$\sigma\f$; variance of the Gaussian.
   */
  AdditiveGaussianPdf(const real sigma);

  /**
   * Construct multivariate pdf.
   *
   * @param sigma \f$\Sigma\f$; covariance of the Gaussian.
   */
  AdditiveGaussianPdf(const M1& sigma);

  /**
   * @copydoc concept::ConditionalPdf::sample()
   */
  template<class V1, class V2>
  void sample(Random& rng, const V1& x1, V2& x2);

  /**
   * @copydoc concept::ConditionalPdf::operator()()
   */
  template<class V1, class V2>
  real operator()(const V1& x1, const V2& x2);

private:
  #ifndef __CUDACC__
  /**
   * Serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};
}

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf() {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(
    const real sigma) : GaussianPdf<V1,M1>(sigma) {
  //
}

template<class V1, class M1>
bi::AdditiveGaussianPdf<V1,M1>::AdditiveGaussianPdf(
    const M1& sigma) : GaussianPdf<V1,M1>(sigma) {
  //
}

template<class V1, class M1>
template<class V1, class V2>
inline void bi::AdditiveGaussianPdf<V1,M1>::sample(Random& rng, const V1& x1,
    V2& x2) {
  /* pre-condition */
  assert(x1.size() == x2.size());

  sample(rng, x2);
  axpy(1.0, x1, x2);
}

template<class V1, class M1>
template<class V1, class V2>
inline real bi::AdditiveGaussianPdf<V1,M1>::operator()(const V1& x1,
    const V2& x2) {
  /* pre-condition */
  assert(x1.size() == x2.size());

  BOOST_AUTO(z, temp_vector<V2>(x2.size()));
  *z = x1;
  axpy(1.0, x2, *z);
  real result = this->operator()(*z);
  delete z;

  return result;
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::AdditiveGaussianPdf<V1,M1>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<GaussianPdf<V1,M1> >(*this);
}
#endif

#endif
