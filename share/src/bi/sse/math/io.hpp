/**
 * @file
 *
 * IO functions for SSE types.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 3055 $
 * $Date: 2012-09-06 17:39:02 +0800 (Thu, 06 Sep 2012) $
 */
#ifndef BI_SSE_MATH_IO_HPP
#define BI_SSE_MATH_IO_HPP

#include "scalar.hpp"

#include <iostream>

/**
 * Output SSE value.
 *
 * @param X Host matrix.
 */
std::ostream& operator<<(std::ostream& stream, const bi::sse_real& x);

inline std::ostream& operator<<(std::ostream& stream, const bi::sse_real& x) {
  stream << '[' << x.unpacked.a;
  stream << ',' << x.unpacked.b;
  #ifdef ENABLE_SINGLE
  stream << ',' << x.unpacked.c;
  stream << ',' << x.unpacked.d;
  #endif
  stream << ']';

  return stream;
}

#endif
