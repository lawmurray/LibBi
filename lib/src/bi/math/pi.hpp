/**
 * @file
 *
 * Common \f$\pi\f$ pre-calculations using big-decimal. We should note that,
 * actually, these are no more accurate than using double precision, but they
 * do at least ensure decent accuracy before casting to single precision.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_PI_HPP
#define BI_MATH_PI_HPP

/**
 * Value of \f$\pi\f$
 */
#define BI_PI 3.1415926535897932384626433832795

/**
 * Value of \f$2\pi\f$
 */
#define BI_TWO_PI 6.2831853071795864769252867665590

/**
 * Value of \f$\sqrt{2\pi}\f$
 */
#define BI_SQRT_TWO_PI 2.5066282746310005024157652848110

/**
 * Value of \f$\frac{1}{2}\ln 2\pi = \ln \sqrt{2\pi}\f$
 */
#define BI_HALF_LOG_TWO_PI 0.91893853320467274178032973640562

#endif
