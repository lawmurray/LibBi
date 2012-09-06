/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/AlmostGaussianKernel.hpp
 */
#ifndef BI_PDF_FASTGAUSSIANKERNEL_HPP
#define BI_PDF_FASTGAUSSIANKERNEL_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Gaussian kernel with 2-norm.
 *
 * @ingroup kd
 *
 * The kernel takes the form:
 *
 * \f[
 *   \mathcal{K}(\mathbf{x}) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}\|\mathbf{x}\|_2^2}
 * \f]
 *
 * The square root and square in the exponent cancel, and so are not
 * computed explicitly.
 *
 * @section Concepts
 *
 * #concept::Kernel
 *
 * @see #hopt for guidance as to bandwidth selection.
 */
class FastGaussianKernel {
public:
  /**
   * Constructor.
   *
   * @param N \f$N\f$; dimensionality of the problem.
   * @param h \f$h\f$; the scaling parameter (bandwidth).
   *
   * Although the kernel itself is not intrinsically dependent on \f$N\f$
   * and \f$h\f$, its normalisation is. Supplying these allows substantial
   * performance increases through precalculation.
   */
  FastGaussianKernel(const int N, const real h);

  /**
   * @copydoc concept::Kernel::bandwidth()
   */
  real bandwidth() const;

  /**
   * @copydoc concept::Kernel::logDensity()
   */
  template<class V1>
  typename V1::value_type logDensity(const V1 x) const;

  /**
   * @copydoc concept::Kernel::density()
   */
  template<class V1>
  typename V1::value_type density(const V1 x) const;

  /**
   * @copydoc concept::Kernel::operator()()
   */
  template<class V1>
  typename V1::value_type operator()(const V1 x) const;

private:
  /**
   * \f$h\f$; bandwidth.
   */
  real h;

  /**
   * \f$(h\sqrt{2\pi})^{-1}\f$; the inverse of the normalisation term.
   */
  real ZI;

  /**
   * \f$\log (h\sqrt{2\pi})\f$; the logarithm of the normalisation term.
   */
  real logZ;

  /**
   * \f$(-2h^2)^{-1}\f$; the exponent term.
   */
  real E;
};
}

inline bi::FastGaussianKernel::FastGaussianKernel(const int N,
    const real h) {
  this->h = h;
  this->ZI = 1.0/(h*BI_SQRT_TWO_PI);
  this->logZ = bi::log(h) + BI_HALF_LOG_TWO_PI;
  this->E = -1.0/(2.0*h*h);
}

inline real bi::FastGaussianKernel::bandwidth() const {
  return h;
}

template<class V1>
inline typename V1::value_type bi::FastGaussianKernel::logDensity(const V1 x) const {
  return E*dot(x) - logZ;
}

template<class V1>
inline typename V1::value_type bi::FastGaussianKernel::density(const V1 x) const {
  typename V1::value_type d = dot(x);
  return ZI*bi::exp(E*d);
}

template<class V1>
inline typename V1::value_type bi::FastGaussianKernel::operator()(const V1 x) const {
  return density(x);
}

#endif
