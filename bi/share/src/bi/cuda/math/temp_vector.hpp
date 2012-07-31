/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_TEMP_VECTOR_HPP
#define BI_CUDA_MATH_TEMP_VECTOR_HPP

#include "vector.hpp"
#include "../../primitive/device_allocator.hpp"
#include "../../primitive/pooled_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

namespace bi {
/**
 * Temporary vector on device.
 *
 * @ingroup math_matvec
 *
 * @tparam T Scalar type.
 *
 * temp_gpu_vector is a convenience class for producing vectors in device
 * memory that are suitable for short-term use before destruction. It uses
 * pooled_allocator to reuse allocated buffers, as device memory allocations
 * can be slow.
 */
template<class T>
struct temp_gpu_vector {
  /**
   * @internal
   *
   * Allocator type.
   */
  typedef pipelined_allocator<pooled_allocator<device_allocator<T> > > allocator_type;

  /**
   * Vector type.
   */
  typedef gpu_vector<T,allocator_type> type;
};
}

#endif
