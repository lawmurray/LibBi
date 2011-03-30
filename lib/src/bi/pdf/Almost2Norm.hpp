/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/Almost2Norm.hpp
 */
#ifndef BI_PDF_ALMOST2NORM_HPP
#define BI_PDF_ALMOST2NORM_HPP

namespace bi {
/**
 * Vector 2-norm for combination with AlmostGaussianKernel.
 *
 * @ingroup math_pdf
 *
 * Almost2Norm is not strictly a norm, as it does not satisfy the property
 * of scalar multiplication. Combining with AlmostGaussianKernel, however,
 * produces the same result as using PNorm<2> and GaussianKernel, but is
 * much more efficient, as the square root in the norm and square in the
 * exponent of the Gaussian are cancelled.
 *
 * @section Concepts
 *
 * #concept::Norm
 *
 * @see AlmostGaussianKernel
 */
class Almost2Norm {
public:
  /**
   * @copydoc concept::Norm::operator()()
   */
  template<class V1>
  typename V1::value_type operator()(const V1& x) const;
};

}

template<class V1>
inline typename V1::value_type bi::Almost2Norm::operator()(const V1& x)
    const {
  return dot(x,x);
}

#endif
