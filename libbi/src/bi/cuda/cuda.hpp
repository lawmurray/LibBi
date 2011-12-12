/**
 * @file
 *
 * Macros for CUDA usage.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_CUDA_HPP
#define BI_CUDA_CUDA_HPP

#include "../misc/assert.hpp"

#include "cuda_runtime.h"

#ifdef __CUDACC__
#include "math_functions.h"
#else
#include "math.h"
#endif

/*
 * CUDA keyword aliases.
 */
#ifdef __CUDACC__
#define CUDA_FUNC_DEVICE __device__
#define CUDA_FUNC_GLOBAL __global__
#define CUDA_FUNC_HOST __host__
#define CUDA_FUNC_BOTH __host__ __device__
#define CUDA_VAR_DEVICE __device__
#define CUDA_VAR_CONSTANT __constant__
#define CUDA_VAR_SHARED __shared__
#define CUDA_NO_INLINE __noinline__
#define CUDA_ALIGN(n) __align__(n)
#else
#define CUDA_FUNC_DEVICE
#define CUDA_FUNC_GLOBAL
#define CUDA_FUNC_HOST
#define CUDA_FUNC_BOTH
#define CUDA_VAR_DEVICE
#define CUDA_VAR_CONSTANT
#define CUDA_VAR_SHARED
#define CUDA_NO_INLINE
#define CUDA_ALIGN(n)
#endif

/**
 * Pitch limit for 2d allocations.
 */
#define CUDA_PITCH_LIMIT 262144

/*
 * Function aliases.
 */
#ifdef USE_DOUBLE
#define CUDA_ABS(x) fabs(x)
#define CUDA_LOG(x) log(x)
#define CUDA_NANLOG(x) (isnan(x) ? log(0.0) : log(x))
#define CUDA_EXP(x) exp(x)
#define CUDA_NANEXP(x) (isnan(x) ? 0.0 : exp(x))
#define CUDA_MAX(x,y) fmax(x,y)
#define CUDA_MIN(x,y) fmin(x,y)
#define CUDA_SQRT(x) sqrt(x)
#define CUDA_POW(x,y) pow(x,y)
#define CUDA_FMOD(x,y) fmod(x,y)
#define CUDA_MODF(x,y) modf(x,y)
#define CUDA_CEIL(x) ceil(x)
#define CUDA_FLOOR(x) floor(x)
#define CUDA_TGAMMA(x) tgamma(x)
#define CUDA_LGAMMA(x) lgamma(x)
#define CUDA_COS(x) cos(x)
#define CUDA_ACOS(x) acos(x)
#define CUDA_SIN(x) sin(x)
#define CUDA_ASIN(x) asin(x)
#else
#define CUDA_ABS(x) fabsf(x)
#define CUDA_LOG(x) logf(x)
#define CUDA_NANLOG(x) (isnan(x) ? logf(0.0f) : logf(x))
#define CUDA_EXP(x) expf(x)
#define CUDA_NANEXP(x) (isnan(x) ? 0.0f : expf(x))
#define CUDA_MAX(x,y) fmaxf(x,y)
#define CUDA_MIN(x,y) fminf(x,y)
#define CUDA_SQRT(x) sqrtf(x)
#define CUDA_POW(x,y) powf(x,y)
#define CUDA_FMOD(x,y) fmodf(x,y)
#define CUDA_MODF(x,y) modff(x,y)
#define CUDA_CEIL(x) ceilf(x)
#define CUDA_FLOOR(x) floorf(x)
#define CUDA_TGAMMA(x) tgammaf(x)
#define CUDA_LGAMMA(x) lgammaf(x)
#define CUDA_COS(x) cosf(x)
#define CUDA_ACOS(x) acosf(x)
#define CUDA_SIN(x) sinf(x)
#define CUDA_ASIN(x) asinf(x)
#endif

/**
 * @def CUDA_SET_CONSTANT(T, x, val)
 *
 * Set a constant on the device.
 *
 * @arg T Constant type.
 * @arg x The constant.
 * @arg val Value to which to set the constant.
 */
#define CUDA_SET_CONSTANT(T, x, val) {\
  CUDA_CHECKED_CALL(cudaMemcpyToSymbol(x, &val, sizeof(T), 0, \
      cudaMemcpyHostToDevice)); \
}

/**
 * @def CUDA_SET_CONSTANT_ASYNC(T, x, val)
 *
 * Set a constant on the device, asynchronously.
 *
 * @arg T Constant type.
 * @arg x The constant.
 * @arg val Value to which to set the constant.
 *
 * @warning The caller should ensure that @p val does not go out of scope
 * before the copy may actually take place.
 */
#define CUDA_SET_CONSTANT_ASYNC(T, x, val) {\
  CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(x, &val, sizeof(T), 0, \
      cudaMemcpyHostToDevice, 0)); \
}

/**
 * @def CUDA_CHECKED_CALL(call)
 *
 * Make CUDA function call and assert success.
 *
 * @arg Function call.
 */
#ifndef NDEBUG
#define CUDA_CHECKED_CALL(call) \
  { \
    cudaError_t cudaErr; \
    cudaErr = call; \
    BI_ASSERT(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#else
#define CUDA_CHECKED_CALL(call) call
#endif

/**
 * @def CUDA_CHECK
 *
 * Check for CUDA errors and fail if detected.
 */
#ifndef NDEBUG
#define CUDA_CHECK \
  { \
    cudaError_t cudaErr; \
    cudaErr = cudaGetLastError(); \
    BI_ASSERT(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#else
#define CUDA_CHECK
#endif

/**
 * Synchronize on device. Corresponds to a call to cudaThreadSynchronize().
 *
 * @note Does nothing if USE_CPU defined.
 */
namespace bi {
  /**
   * Synchronize with device.
   */
  inline void synchronize() {
    #ifndef USE_CPU
    CUDA_CHECKED_CALL(cudaThreadSynchronize());
    #endif
  }

  /**
   * Synchronize with device.
   *
   * @param stream Stream with which to synchronize.
   */
  inline void synchronize(cudaStream_t stream) {
    #ifndef USE_CPU
    CUDA_CHECKED_CALL(cudaStreamSynchronize(stream));
    #endif
  }
}

#endif
