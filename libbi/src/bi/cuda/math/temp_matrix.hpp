/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_TEMP_MATRIX_HPP
#define BI_CUDA_MATH_TEMP_MATRIX_HPP

#include "matrix.hpp"
#include "../../primitive/device_allocator.hpp"
#include "../../primitive/pooled_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

namespace bi {
/**
 * Temporary matrix on device.
 *
 * @ingroup math_matvec
 *
 * @tparam T Scalar type.
 *
 * temp_gpu_matrix is a convenience class for producing matrices in device
 * memory that are suitable for short-term use before destruction. It uses
 * pooled_allocator to reuse allocated buffers, as device memory allocations
 * can be slow.
 */
template<class T>
struct temp_gpu_matrix {
  /**
   * @internal
   *
   * Allocator type.
   */
  typedef pipelined_allocator<pooled_allocator<device_allocator<T> > > allocator_type;

  /**
   * matrix type.
   */
  typedef gpu_matrix<T,allocator_type> type;
};
}

#endif
