/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_DEVICEALLOCATOR_HPP
#define BI_PRIMITIVE_DEVICEALLOCATOR_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Allocator for device memory.
 *
 * @ingroup primitive_allocator
 */
template<class T>
class device_allocator {
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
    typedef device_allocator<U> other;
  };

  device_allocator() {
    //
  }

  template<class U>
  device_allocator(const device_allocator<U>& o) {
    //
  }

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
    CUDA_CHECKED_CALL(cudaMalloc((void**)&ptr, num*sizeof(T)));
    return ptr;
  }

  void construct(pointer p, const T& t) {
    new ((void*)p) T(t);
  }

  void destroy(pointer p) {
    ((T*)p)->~T();
  }

  void deallocate(pointer p, size_type num) {
    CUDA_CHECKED_CALL(cudaFree(p));
  }

  template<class U>
  bool operator==(const device_allocator<U>& o) const {
    return true;
  }

  template<class U>
  bool operator!=(const device_allocator<U>& o) const {
    return false;
  }

};

}

#endif
