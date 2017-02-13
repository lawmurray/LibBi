/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_CURANDSTATESA_HPP
#define BI_CUDA_RANDOM_CURANDSTATESA_HPP

#include "../cuda.hpp"
#include "../device.hpp"

#ifdef ENABLE_CUDA
#include "curand_kernel.h"
#include "curand_poisson.h"
#endif

namespace bi {
/**
 * Struct-of-arrays storage of multiple objects of type curandStateXORWOW,
 * from curand_kernel.h. This allows coalesced memory access when loading and
 * storing device random number generators.
 */
class curandStateSA {
public:
  /**
   * Constructor.
   */
  curandStateSA();

  /**
   * Copy constructor.
   */
  curandStateSA(const curandStateSA& o);

  /**
   * Destructor.
   */
  ~curandStateSA();

  /**
   * Load state from arrays.
   *
   * @param i Index of state to load.
   * @param[out] r State.
   */
  CUDA_FUNC_DEVICE
  void load(const int i, curandState& r) const;

  /**
   * Store state into arrays.
   *
   * @param i Index of state to store.
   * @param r State.
   */
  CUDA_FUNC_DEVICE
  void store(const int i, const curandState& r);

private:
  unsigned int *d, *v0, *v1, *v2, *v3, *v4;
  int *boxmuller_flag, *boxmuller_flag_double;
  float* boxmuller_extra;
  double* boxmuller_extra_double;
  bool own;
};
}

inline bi::curandStateSA::curandStateSA() :
    own(true) {
  const int n = deviceIdealThreads();

  CUDA_CHECKED_CALL(cudaMalloc(&d, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&v0, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&v1, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&v2, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&v3, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&v4, n*sizeof(unsigned int)));
  CUDA_CHECKED_CALL(cudaMalloc(&boxmuller_flag, n*sizeof(int)));
  CUDA_CHECKED_CALL(cudaMalloc(&boxmuller_flag_double, n*sizeof(int)));
  CUDA_CHECKED_CALL(cudaMalloc(&boxmuller_extra, n*sizeof(float)));
  CUDA_CHECKED_CALL(cudaMalloc(&boxmuller_extra_double, n*sizeof(double)));
}

inline bi::curandStateSA::curandStateSA(const curandStateSA& o) :
    d(o.d), v0(o.v0), v1(o.v1), v2(o.v2), v3(o.v3), v4(o.v4), boxmuller_flag(
        o.boxmuller_flag), boxmuller_flag_double(o.boxmuller_flag_double), boxmuller_extra(
        o.boxmuller_extra), boxmuller_extra_double(o.boxmuller_extra_double), own(
        false) {
  //
}

inline bi::curandStateSA::~curandStateSA() {
  if (own) {
    CUDA_CHECKED_CALL(cudaFree(d));
    CUDA_CHECKED_CALL(cudaFree(v0));
    CUDA_CHECKED_CALL(cudaFree(v1));
    CUDA_CHECKED_CALL(cudaFree(v2));
    CUDA_CHECKED_CALL(cudaFree(v3));
    CUDA_CHECKED_CALL(cudaFree(v4));
    CUDA_CHECKED_CALL(cudaFree(boxmuller_flag));
    CUDA_CHECKED_CALL(cudaFree(boxmuller_flag_double));
    CUDA_CHECKED_CALL(cudaFree(boxmuller_extra));
    CUDA_CHECKED_CALL(cudaFree(boxmuller_extra_double));
  }
}

inline void bi::curandStateSA::load(const int i, curandState& r) const {
  r.d = d[i];
  r.v[0] = v0[i];
  r.v[1] = v1[i];
  r.v[2] = v2[i];
  r.v[3] = v3[i];
  r.v[4] = v4[i];
  r.boxmuller_flag = boxmuller_flag[i];
  r.boxmuller_flag_double = boxmuller_flag_double[i];
  r.boxmuller_extra = boxmuller_extra[i];
  r.boxmuller_extra_double = boxmuller_extra_double[i];
}

inline void bi::curandStateSA::store(const int i, const curandState& r) {
  d[i] = r.d;
  v0[i] = r.v[0];
  v1[i] = r.v[1];
  v2[i] = r.v[2];
  v3[i] = r.v[3];
  v4[i] = r.v[4];
  boxmuller_flag[i] = r.boxmuller_flag;
  boxmuller_flag_double[i] = r.boxmuller_flag_double;
  boxmuller_extra[i] = r.boxmuller_extra;
  boxmuller_extra_double[i] = r.boxmuller_extra_double;
}

#endif
