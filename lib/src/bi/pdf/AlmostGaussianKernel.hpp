/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/AlmostGaussianKernel.hpp
 */
#ifndef BI_PDF_ALMOSTGAUSSIANKERNEL_HPP
#define BI_PDF_ALMOSTGAUSSIANKERNEL_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Gaussian kernel for combination with Almost2Norm.
 *
 * @ingroup math_pdf
 *
 * The kernel takes the form:
 *
 * \f[
 *   \mathcal{K}(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x}
 * \f]
 *
 * Note that the \f$x\f$ in the exponent is not squared as per the usual
 * Gaussian. This means that the kernel is actually a scaled Laplacian
 * (i.e. does not integrate to 1). Combining with Almost2Norm, however,
 * produces the same result as using PNorm<2> and GaussianKernel, but is
 * much more efficient, as the square root in the norm and square in the
 * exponent of the Gaussian are cancelled.
 *
 * @section Concepts
 *
 * #concept::Kernel
 *
 * @see #hopt for guidance as to bandwidth selection.
 */
class AlmostGaussianKernel {
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
  AlmostGaussianKernel(const int N, const real h);

  /**
   * @copydoc concept::Kernel::operator()()
   */
  template<class T1>
  T1 operator()(const T1 x) const;

private:
  /**
   * \f$(h\sqrt{2\pi})^{-1}\f$; the normalisation term.
   */
  real ZI;

  /**
   * \f$(-2h^2)^{-1}\f$; the exponent term.
   */
  real E;
};
}

inline bi::AlmostGaussianKernel::AlmostGaussianKernel(const int N,
    const real h) {
  ZI = 1.0/pow(h*BI_SQRT_TWO_PI, N);
  E = -1.0/(2.0*h*h);
}

template<class T1>
inline T1 bi::AlmostGaussianKernel::operator()(const T1 x)
    const {
  return ZI*exp(E*x);
}

#endif
