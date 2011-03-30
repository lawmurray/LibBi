/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "omp.hpp"

BI_THREAD int bi_omp_tid;
BI_THREAD int bi_omp_max_threads;

void bi_omp_init() {
  /* explicitly turn off dynamic threads, required for threadprivate
   * guarantees */
  omp_set_dynamic(0);

  int max_threads = omp_get_max_threads(); // must be outside parallel block
  #pragma omp parallel
  {
    bi_omp_tid = omp_get_thread_num();
    bi_omp_max_threads = max_threads;
  }
}
