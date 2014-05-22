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

#ifdef ENABLE_CUDA
#include "cuda_runtime.h"
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
#ifndef ENABLE_CUDA
#define CUDA_CHECKED_CALL(call)
#elif !defined(NDEBUG)
#define CUDA_CHECKED_CALL(call) \
  { \
    cudaError_t cudaErr; \
    cudaErr = call; \
    cudaThreadSynchronize(); \
    BI_ASSERT_MSG(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#else
#define CUDA_CHECKED_CALL(call) \
  { \
    cudaError_t cudaErr; \
    cudaErr = call; \
    BI_ERROR_MSG(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#endif

/**
 * @def CUDA_CHECK
 *
 * Check for CUDA errors and fail if detected.
 */
#ifndef ENABLE_CUDA
#define CUDA_CHECK
#elif !defined(NDEBUG)
#define CUDA_CHECK \
  { \
    cudaError_t cudaErr; \
    cudaThreadSynchronize(); \
    cudaErr = cudaGetLastError(); \
    BI_ASSERT_MSG(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#else
#define CUDA_CHECK \
  { \
    cudaError_t cudaErr; \
    cudaErr = cudaGetLastError(); \
    BI_ERROR_MSG(cudaErr == cudaSuccess, cudaGetErrorString(cudaErr)); \
  }
#endif

namespace bi {
  /**
   * Synchronize with device.
   *
   * @param sync True to synchronize, false if not.
   */
  inline void synchronize(const bool sync = true) {
    #ifdef ENABLE_CUDA
    if (sync) {
      CUDA_CHECKED_CALL(cudaThreadSynchronize());
    }
    #endif
  }

  #ifdef ENABLE_CUDA
  /**
   * Synchronize with device.
   *
   * @param stream Stream with which to synchronize.
   * @param sync True to synchronize, false if not.
   */
  inline void synchronize(cudaStream_t stream, const bool sync = true) {
    if (sync) {
      CUDA_CHECKED_CALL(cudaStreamSynchronize(stream));
    }
  }
  #endif
}

#endif
