/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_TEMPMATRIX_HPP
#define BI_HOST_MATH_TEMPMATRIX_HPP

#include "matrix.hpp"
#include "../../primitive/pinned_allocator.hpp"
#include "../../primitive/aligned_allocator.hpp"
#include "../../primitive/pooled_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

namespace bi {
/**
 * Temporary matrix on host.
 *
 * @ingroup math_matvec
 *
 * @tparam VM1 Vector or matrix type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 *
 * temp_host_matrix is a convenience class for producing matrices in main
 * memory that are suitable for short-term use before destruction. It uses
 * pooled_allocator to reuse allocated buffers, and when GPU devices
 * are enabled, pinned_allocator for faster copying between host and device.
 */
template<class T, int size1_value = -1, int size2_value = -1, int lead_value =
    -1, int inc_value = 1>
struct temp_host_matrix {
  /**
   * Allocator type.
   *
   * Note that, on host, pooled_allocator is slower than std::allocator. It
   * is in avoiding calls to pinned_allocator (which internally calls
   * cudaMallocHost) where there are some performance gains.
   */
  #ifdef ENABLE_CUDA
  typedef pipelined_allocator<pooled_allocator<pinned_allocator<T> > > allocator_type;
  #else
  typedef pooled_allocator<aligned_allocator<T> > allocator_type;
  #endif

  /**
   * Matrix type.
   */
  typedef host_matrix<T,size1_value,size2_value,lead_value,inc_value,
      allocator_type> type;
};
}

#endif
