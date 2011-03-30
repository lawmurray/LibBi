/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef POOLED_ALLOCATOR_HPP
#define POOLED_ALLOCATOR_HPP

#include "assert.hpp"

#include <map>

namespace bi {
/**
 * Wraps another allocator to provide reusable pool of allocations. Not
 * thread safe.
 *
 * @tparam A Other allocator type.
 *
 * @ingroup misc_allocators
 */
template<class A>
class pooled_allocator {
public:
  typedef typename A::size_type size_type;
  typedef typename A::difference_type difference_type;
  typedef typename A::pointer pointer;
  typedef typename A::const_pointer const_pointer;
  typedef typename A::reference reference;
  typedef typename A::const_reference const_reference;
  typedef typename A::value_type value_type;

  template <class U>
  struct rebind {
    typedef pooled_allocator<typename A::template rebind<U>::other> other;
  };

  pooled_allocator() {
    //
  }

  pooled_allocator(const pooled_allocator<A>& o) {
    //
  }

  ~pooled_allocator();

  pointer address(reference value) const;

  const_pointer address(const_reference value) const;

  size_type max_size() const;

  pointer allocate(size_type num, const_pointer *hint = 0);

  void construct(pointer p, const value_type& t);

  void destroy(pointer p);

  void deallocate(pointer p, size_type num);

  bool operator==(const pooled_allocator<A>& o) const {
    return true;
  }

  template<class U>
  bool operator==(const pooled_allocator<U>& o) const {
    return false;
  }

  bool operator!=(const pooled_allocator<A>& o) const {
    return false;
  }

  template<class U>
  bool operator!=(const pooled_allocator<U>& o) const {
    return true;
  }

private:
  /**
   * Wrapped allocator.
   */
  A alloc;

  /**
   * Full pool, mapping pointers to sizes.
   */
  static std::map<pointer,size_type> pool;

  /**
   * Available items in pool, mapping sizes to pointers.
   */
  static std::multimap<size_type,pointer> available;

};

}

template<class A>
std::map<typename A::pointer,typename A::size_type> bi::pooled_allocator<A>::pool;

template<class A>
std::multimap<typename A::size_type,typename A::pointer> bi::pooled_allocator<A>::available;

template<class A>
bi::pooled_allocator<A>::~pooled_allocator() {
  /* clean up pool */
//  typename std::map<pointer,size_type>::iterator iter;
//  for (iter = pool.begin(); iter != pool.end(); ++iter) {
//    alloc.deallocate(iter->first, iter->second);
//  }
}

template<class A>
inline typename bi::pooled_allocator<A>::pointer
    bi::pooled_allocator<A>::address(reference value) const {
  return alloc.address(value);
};

template<class A>
inline typename bi::pooled_allocator<A>::const_pointer
    bi::pooled_allocator<A>::address(const_reference value) const {
  return alloc.address(value);
};

template<class A>
inline typename bi::pooled_allocator<A>::size_type
    bi::pooled_allocator<A>::max_size() const {
  return alloc.max_size();
};

template<class A>
typename bi::pooled_allocator<A>::pointer bi::pooled_allocator<A>::allocate(
    size_type num, const_pointer *hint) {
  pointer p;

  /* check available items to reuse */
  typename std::multimap<size_type,pointer>::iterator iter;
  iter = available.lower_bound(num);
  if (iter != available.end()) {
    /* reuse item */
    p = iter->second;
    available.erase(iter);
  } else {
    /* new item */
    p = alloc.allocate(num, hint);
    pool.insert(std::make_pair(p, num));
  }
  return p;
}

template<class A>
inline void bi::pooled_allocator<A>::construct(pointer p, const value_type& t) {
   alloc.construct(p, t);
}

template<class A>
inline void bi::pooled_allocator<A>::destroy(pointer p) {
  alloc.destroy(p);
}

template<class A>
void bi::pooled_allocator<A>::deallocate(pointer p, size_type num) {
  /* return to pool rather than deallocating */
  typename std::map<pointer,size_type>::iterator iter = pool.find(p);
  assert (iter != pool.end());
  available.insert(std::make_pair(iter->second, iter->first));
}

#endif
