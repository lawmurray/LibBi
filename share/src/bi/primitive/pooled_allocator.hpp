/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_POOLEDALLOCATOR_HPP
#define BI_PRIMITIVE_POOLEDALLOCATOR_HPP

#include "../misc/omp.hpp"
#include "../misc/assert.hpp"

#include <map>
#include <list>
#include <vector>

namespace bi {
/**
 * Wraps another allocator to provide reusable pool of allocations.
 *
 * @tparam A Other allocator type.
 *
 * @ingroup primitive_allocator
 *
 * This class is thread safe.
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

  /**
   * Allocate new item, drawing from pool if possible.
   */
  pointer allocate(size_type num, const_pointer *hint = 0);

  void construct(pointer p, const value_type& t);

  void destroy(pointer p);

  /**
   * Return item to pool.
   */
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

  /**
   * Empty pool.
   */
  void empty();

private:
  /**
   * Pool type.
   *
   * One list is constructed per size. Reused allocations are drawn from and
   * returned to the end of each list
   *
   * Buffers are removed from and returned to the end of each
   *
   *
   * @note The type used here, lists indexed by sizes, seems substantially
   * faster than multimap.
   */
  typedef std::map<size_type, std::list<pointer> > pool_type;

  /**
   * Wrapped allocator.
   */
  A alloc;

  /**
   * Available items in pools, mapping sizes to pointers, indexed by thread.
   */
  static std::vector<pool_type> available;
};

}

#include "boost/typeof/typeof.hpp"

template<class A>
std::vector<typename bi::pooled_allocator<A>::pool_type> bi::pooled_allocator<A>::available;

template<class A>
inline bi::pooled_allocator<A>::~pooled_allocator() {
  //
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
inline typename bi::pooled_allocator<A>::pointer bi::pooled_allocator<A>::allocate(
    size_type num, const_pointer *hint) {
  pointer p;

  /* init if necessary */
  if (bi_omp_max_threads > (int)available.size()) {
    /* this outer conditional avoids the critical section most the time, but
     * multiple threads may get this far */
    #pragma omp critical
    {
      if (bi_omp_max_threads > (int)available.size()) {
        /* only one thread gets this far */
        available.resize(bi_omp_max_threads);
      }
    }
  }

  /* check available items to reuse */
  if (num > 0) {
    BOOST_AUTO(iter, available[bi_omp_tid].find(num));
    /* ^ can use lower_bound() to get buffer of *at least* size num, but will
     * be returned to pool as if size num, not >= num, so find() is used to get
     * buffers only of exactly size num instead. */
    if (iter != available[bi_omp_tid].end()) {
      /* existing item */
      BI_ASSERT(!iter->second.empty());
      p = iter->second.back();
      iter->second.pop_back();
      if (iter->second.empty()) {
        available[bi_omp_tid].erase(iter);
      }
    } else {
      /* new item */
      p = alloc.allocate(num, hint);
    }
  } else {
    p = NULL;
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
inline void bi::pooled_allocator<A>::deallocate(pointer p, size_type num) {
  if (p != NULL) {
    /* return to pool for reuse; note insert won't insert if key already
     * exists, but in either case returns iterator for that key */
    available[bi_omp_tid].insert(std::make_pair(num,
        std::list<pointer>())).first->second.push_back(p);
  } else {
    alloc.deallocate(p, num);
  }
}

template<class A>
inline void bi::pooled_allocator<A>::empty() {
  BOOST_AUTO(iter1, available[bi_omp_tid].begin());
  for (; iter1 != available[bi_omp_tid].end(); ++iter1) {
    BOOST_AUTO(iter2, iter1->second.begin());
    for (; iter2 != iter1->second.end(); ++iter2) {
      alloc.deallocate(*iter2, iter1->first);
    }
  }
  available[bi_omp_tid].clear();
}

#endif
