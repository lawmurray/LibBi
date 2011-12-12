/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "omp.hpp"

#include "../cuda/cuda.hpp"

BI_THREAD int bi_omp_tid;
BI_THREAD int bi_omp_max_threads;
BI_THREAD cublasHandle_t bi_omp_cublas_handle;
BI_THREAD cudaStream_t bi_omp_cuda_stream;

void bi_omp_init() {
  /* explicitly turn off dynamic threads, required for threadprivate
   * guarantees */
  omp_set_dynamic(0);

  /* allow nested parallelism */
  omp_set_nested(1);

  int max_threads = omp_get_max_threads(); // must be outside parallel block
  #pragma omp parallel
  {
    bi_omp_tid = omp_get_thread_num();
    bi_omp_max_threads = max_threads;
    #ifndef USE_CPU
    CUBLAS_CHECKED_CALL(cublasCreate(&bi_omp_cublas_handle));
    CUDA_CHECKED_CALL(cudaStreamCreate(&bi_omp_cuda_stream));
    #endif
  }
}

void bi_omp_term() {
  #pragma omp parallel
  {
    #ifndef USE_CPU
    CUBLAS_CHECKED_CALL(cublasDestroy(bi_omp_cublas_handle));
    CUDA_CHECKED_CALL(cudaStreamDestroy(bi_omp_cuda_stream));
    #endif
  }
}
