/**
 * @file
 *
 * Types and operators for Streaming SIMD Extensions (SSE).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_SCALAR_HPP
#define BI_SSE_MATH_SCALAR_HPP

#include "../../math/scalar.hpp"

#ifdef ENABLE_SSE
#include "sse_float.hpp"
#include "sse_double.hpp"
#endif

#ifdef ENABLE_AVX
#include "avx_float.hpp"
#include "avx_double.hpp"
#endif

namespace bi {
#if defined(ENABLE_SINGLE) && defined(ENABLE_AVX)
typedef avx_float simd_real;
#elif defined(ENABLE_SINGLE) && defined(ENABLE_SSE)
typedef sse_float simd_real;
#elif defined(ENABLE_AVX)
typedef avx_double simd_real;
#elif defined(ENABLE_SSE)
typedef sse_double simd_real;
#else
typedef real simd_real;
#endif
}

#define BI_SIMD_SIZE (sizeof(simd_real)/sizeof(real))

#endif
