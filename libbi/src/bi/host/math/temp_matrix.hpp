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
#include "../../primitive/pooled_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

namespace bi {
/**
 * Temporary matrix on host.
 *
 * @ingroup math_matvec
 *
 * temp_host_matrix is a convenience class for producing matrices in main
 * memory that are suitable for short-term use before destruction. It uses
 * pooled_allocator to reuse allocated buffers, and when GPU devices
 * are enabled, pinned_allocator for faster copying between host and device.
 */
template<class T>
struct temp_host_matrix {
  /**
   * @internal
   *
   * Allocator type.
   */
  typedef pipelined_allocator<pooled_allocator<pinned_allocator<T> > > allocator_type;

  /**
   * matrix type.
   */
  typedef host_matrix<T,allocator_type> type;
};
}

#endif
