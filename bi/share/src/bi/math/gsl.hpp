/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_GSL_HPP
#define BI_MATH_GSL_HPP

#include "vector.hpp"

#include <gsl/gsl_vector.h>

namespace bi {
/**
 * View of GSL vector.
 *
 * @param x The GSL vector.
 */
host_vector_reference<double> gsl_vector_reference(gsl_vector* x);

/**
 * View of GSL vector.
 *
 * @param x The GSL vector.
 */
const host_vector_reference<double> gsl_vector_reference(const gsl_vector* x);

}

bi::host_vector_reference<double> bi::gsl_vector_reference(gsl_vector* x) {
  return host_vector_reference<double>(x->data, x->size, x->stride);
}

const bi::host_vector_reference<double> bi::gsl_vector_reference(
    const gsl_vector* x) {
  return host_vector_reference<double>(x->data, x->size, x->stride);
}

#endif
