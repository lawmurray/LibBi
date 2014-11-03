/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PINNEDALLOCATOR_HPP
#define BI_PRIMITIVE_PINNEDALLOCATOR_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Allocator for pinned host memory. If ENABLE_CUDA is not defined, reverts to
 * pageable host memory.
 *
 * @ingroup primitive_allocator
 */
#ifdef ENABLE_CUDA
template <class T>
class pinned_allocator {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef pinned_allocator<U> other;
  };

  pointer address(reference value) const {
    return &value;
  };

  const_pointer address(const_reference value) const {
    return &value;
  };

  size_type max_size() const {
    return size_type(-1) / sizeof(T);
  };

  pointer allocate(size_type num, const_pointer *hint = 0) {
    pointer ptr;
    CUDA_CHECKED_CALL(cudaMallocHost((void**)&ptr, num*sizeof(T)));
    return ptr;
  }

  void construct(pointer p, const T& t) {
    new ((void*)p) T(t);
  }

  void destroy(pointer p) {
    ((T*)p)->~T();
  }

  void deallocate(pointer p, size_type num) {
    CUDA_CHECKED_CALL(cudaFreeHost(p));
  }

  template<class U>
  bool operator==(const pinned_allocator<U>& o) {
    return true;
  }

  template<class U>
  bool operator!=(const pinned_allocator<U>& o) {
    return false;
  }
};
#else
template<class T>
class pinned_allocator : public std::allocator<T> {
  //
};
#endif
}

#endif
