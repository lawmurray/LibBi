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
#include "../cuda/math/cublas.hpp"

#include "omp.h"

/**
 * @def BI_THREAD_LOCAL_DEC(type, name)
 *
 * Declare thread-local variables.
 *
 * @arg type Variable type.
 * @arg name Variable name.
 */
#define BI_THREAD_LOCAL_DEC(type, name) \
extern BI_THREAD type name;


/**
 * @def BI_THREAD_LOCAL_DEF(type, name)
 *
 * Define thread-local variables.
 *
 * @arg type Variable type.
 * @arg name Variable name.
 */
#define BI_THREAD_LOCAL_DEF(type, name) \
  BI_THREAD type name;

/**
 * Thread id.
 */
BI_THREAD_LOCAL_DEC(int, bi_omp_tid)

/**
 * Maximum number of threads. Saves function calls to omp_get_max_threads().
 */
BI_THREAD_LOCAL_DEC(int, bi_omp_max_threads)

/**
 * CUBLAS context handle for CUBLAS function calls (API v2).
 */
BI_THREAD_LOCAL_DEC(cublasHandle_t, bi_omp_cublas_handle)

/**
 * CUDA stream. Default stream is used unless this explicitly specified. May
 * also be used for CUBLAS calls.
 */
BI_THREAD_LOCAL_DEC(cudaStream_t, bi_omp_cuda_stream)

#ifdef __ICC
#pragma omp threadprivate(bi_omp_tid)
#pragma omp threadprivate(bi_omp_max_threads)
#pragma omp threadprivate(bi_omp_cublas_handle)
#pragma omp threadprivate(bi_omp_cuda_stream)
#endif

/**
 * Initialise OpenMP environment (thread private variables).
 */
void bi_omp_init();

/**
 * Terminate OpenMP environment (thread private variables).
 */
void bi_omp_term();

#endif
