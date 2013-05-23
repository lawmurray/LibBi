/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_ALIGNED_ALLOCATOR_HPP
#define BI_MISC_ALIGNED_ALLOCATOR_HPP

#include <cstdlib>

namespace bi {
/**
 * Allocator for aligned memory. Useful to align buffers for ready loading
 * of SSE 128-bit values.
 *
 * @ingroup primitive_allocators
 */
template <class T, unsigned X = 16>
class aligned_allocator {
public:
  typedef size_t size_type;
  typedef size_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef aligned_allocator<U,X> other;
  };

  aligned_allocator() {
    //
  }

  template<class U, unsigned Y>
  aligned_allocator(const aligned_allocator<U,Y>& o) {
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
    int err = posix_memalign((void**)&ptr, X, num*sizeof(T));
    BI_ASSERT_MSG(err == 0, "Aligned memory allocation failed");
    return ptr;
  }

  void construct(pointer p, const T& t) {
    new ((void*)p) T(t);
  }

  void destroy(pointer p) {
    ((T*)p)->~T();
  }

  void deallocate(pointer p, size_type num) {
    free(p);
  }

  template<class U, unsigned Y>
  bool operator==(const aligned_allocator<U,Y>& o) const {
    return true;
  }

  template<class U, unsigned Y>
  bool operator!=(const aligned_allocator<U,Y>& o) const {
    return false;
  }

};

}

#endif
