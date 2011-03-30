/**
 * @file
 *
 * Utility functions for OpenMP.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_OMP_HPP
#define BI_MISC_OMP_HPP

#include "compile.hpp"

#include "omp.h"

extern BI_THREAD int bi_omp_tid;
extern BI_THREAD int bi_omp_max_threads;

#ifdef __ICC
#pragma omp threadprivate(bi_omp_tid)
#pragma omp threadprivate(bi_omp_max_threads)
#endif

/**
 * Initialise OpenMP environment (thread private variables).
 */
void bi_omp_init();

#endif
