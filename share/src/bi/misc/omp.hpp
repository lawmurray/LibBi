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
#ifdef ENABLE_CUDA
#include "../cuda/cuda.hpp"
#include "../cuda/math/cublas.hpp"
#endif

#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
#include "omp.h"
#endif

/**
 * Thread id.
 */
extern BI_THREAD int bi_omp_tid;

/**
 * Maximum number of threads. Saves function calls to omp_get_max_threads().
 */
extern int bi_omp_max_threads;

#ifdef ENABLE_CUDA
/**
 * CUBLAS context handle for CUBLAS function calls (API v2).
 */
extern BI_THREAD cublasHandle_t bi_omp_cublas_handle;

/**
 * CUDA stream. Default stream is used unless this explicitly specified. May
 * also be used for CUBLAS calls.
 */
extern BI_THREAD cudaStream_t bi_omp_cuda_stream;
#endif

#ifdef __ICC
#pragma omp threadprivate(bi_omp_tid)
#ifdef ENABLE_CUDA
#pragma omp threadprivate(bi_omp_cublas_handle)
#pragma omp threadprivate(bi_omp_cuda_stream)
#endif
#endif

/**
 * Initialise OpenMP environment.
 *
 * @param threads Number of threads. Zero for the default.
 */
void bi_omp_init(const int threads = 0);

/**
 * Terminate OpenMP environment.
 */
void bi_omp_term();

#endif
