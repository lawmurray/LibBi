/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_TEMPVECTOR_HPP
#define BI_HOST_MATH_TEMPVECTOR_HPP

#include "vector.hpp"
#include "../../primitive/pinned_allocator.hpp"
#include "../../primitive/pooled_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

namespace bi {
/**
 * Temporary vector on host.
 *
 * @ingroup math_matvec
 *
 * temp_host_vector is a convenience class for producing vectors in main
 * memory that are suitable for short-term use before destruction. It uses
 * pooled_allocator to reuse allocated buffers, and when GPU devices
 * are enabled, pinned_allocator for faster copying between host and device.
 */
template<class T>
struct temp_host_vector {
  /**
   * @internal
   *
   * Allocator type.
   */
  typedef pipelined_allocator<pooled_allocator<pinned_allocator<T> > > allocator_type;

  /**
   * Vector type.
   */
  typedef host_vector<T,allocator_type> type;
};
}

#endif
