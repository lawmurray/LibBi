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
int bi_omp_max_threads;

#ifdef ENABLE_CUDA
BI_THREAD cublasHandle_t bi_omp_cublas_handle;
BI_THREAD cudaStream_t bi_omp_cuda_stream;
#endif

void bi_omp_init(const int threads) {
  #if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
  /* explicitly turn off dynamic threads, required for threadprivate
   * guarantees */
  omp_set_dynamic(0);

  /* allow nested parallelism */
  //omp_set_nested(1);

  /* use static scheduling for pseudorandom sequence reproducibility */
  omp_set_schedule(omp_sched_static, 0);

  /* set number of threads */
  if (threads > 0) {
    omp_set_num_threads(threads);
  }

  bi_omp_max_threads = omp_get_max_threads(); // must be outside parallel block
  #pragma omp parallel
  {
    bi_omp_tid = omp_get_thread_num();
    #ifdef ENABLE_CUDA
    CUBLAS_CHECKED_CALL(cublasCreate(&bi_omp_cublas_handle));
    CUDA_CHECKED_CALL(cudaStreamCreate(&bi_omp_cuda_stream));
    #endif
  }
#else
  bi_omp_max_threads = 1;
  bi_omp_tid = 0;
  #ifdef ENABLE_CUDA
  CUBLAS_CHECKED_CALL(cublasCreate(&bi_omp_cublas_handle));
  CUDA_CHECKED_CALL(cudaStreamCreate(&bi_omp_cuda_stream));
  #endif
#endif
}

void bi_omp_term() {
  #pragma omp parallel
  {
    #ifdef ENABLE_CUDA
    CUBLAS_CHECKED_CALL(cublasDestroy(bi_omp_cublas_handle));
    CUDA_CHECKED_CALL(cudaStreamDestroy(bi_omp_cuda_stream));
    #endif
  }
}
