/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PIPELINEDALLOCATOR_HPP
#define BI_PRIMITIVE_PIPELINEDALLOCATOR_HPP

#include <vector>
#include <list>
#include <cstdlib>

namespace bi {
/**
 * Wraps an allocator and pipelines its deallocations with respect to a CUDA
 * stream.
 *
 * @ingroup primitive_allocator
 *
 * Every deallocation is deferred and an event added to the CUDA stream
 * associated with the current thread. The deallocation is then not actually
 * performed until after that event has been reached by the CUDA stream.
 *
 * Consider the following code:
 *
 * @code
 * {
 *   gpu_vector<float> dev(10);
 *   host_vector<float> host(10);
 *   ...
 *   dev = host;
 * }
 * @endcode
 *
 * The danger here is that the asynchronous copy on the last line may not
 * have been completed by the time @c host goes out of scope and its buffer
 * is deallocated. This may be corrected using pipelined_allocator:
 *
 * @code
 * {
 *   gpu_vector<float> dev(10);
 *   host_vector<float, pipelined_allocator<std::allocator<float> > > host(10);
 *   ...
 *   dev = host;
 * }
 * @endcode
 *
 * Now, despite @c host going out of scope, its buffer is not actually
 * deallocated until the copy has completed.
 *
 * This class is thread safe.
 */
template<class A>
class pipelined_allocator : public A {
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
    typedef pipelined_allocator<typename A::template rebind<U>::other> other;
  };

  /**
   * Pipelined deallocation.
   */
  void deallocate(pointer p, size_type num);

  /**
   * Initialise if necessary.
   */
  void init();

  /**
   * Check and deallocate as many outstanding buffers as possible.
   */
  void clean();

  /**
   * Add buffer for deferred deallocation.
   */
  void add(pointer buf, size_type num);

  /**
   * Empty pipeline.
   */
  void empty();

private:
  #ifdef ENABLE_CUDA
  /**
   * Buffer lists, indexed by thread.
   */
  static std::vector<std::list<pointer> > bufs;

  /**
   * Buffer size lists, indexed by thread.
   */
  static std::vector<std::list<size_type> > sizes;

  /**
   * CUDA stream event lists, indexed by thread.
   */
  static std::vector<std::list<cudaEvent_t> > evts;
  #endif
};

}

#include "../misc/omp.hpp"

#ifdef ENABLE_CUDA
template<class A>
std::vector<std::list<typename bi::pipelined_allocator<A>::pointer> > bi::pipelined_allocator<A>::bufs;

template<class A>
std::vector<std::list<typename bi::pipelined_allocator<A>::size_type> > bi::pipelined_allocator<A>::sizes;

template<class A>
std::vector<std::list<cudaEvent_t> > bi::pipelined_allocator<A>::evts;
#endif

template<class A>
inline void bi::pipelined_allocator<A>::deallocate(pointer p, size_type num) {
  #ifdef ENABLE_CUDA
  init();
  clean();
  add(p, num);
  #else
  A::deallocate(p, num);
  #endif
}

template<class A>
void bi::pipelined_allocator<A>::init() {
  #ifdef ENABLE_CUDA
  if (bi_omp_max_threads > (int)evts.size()) {
    /* this outer conditional avoids the critical section most the time, but
     * multiple threads may get this far */
    #pragma omp critical
    {
      if (bi_omp_max_threads > (int)evts.size()) {
        /* only one thread gets this far */
        evts.resize(bi_omp_max_threads);
        bufs.resize(bi_omp_max_threads);
        sizes.resize(bi_omp_max_threads);
      }
    }
  }
  #endif
}

template<class A>
void bi::pipelined_allocator<A>::clean() {
  #ifdef ENABLE_CUDA
  BOOST_AUTO(evtIter, evts[bi_omp_tid].begin());
  BOOST_AUTO(bufIter, bufs[bi_omp_tid].begin());
  BOOST_AUTO(sizeIter, sizes[bi_omp_tid].begin());

  bool done = evtIter == evts[bi_omp_tid].end() || cudaEventQuery(*evtIter) != cudaSuccess;
  while (!done) {
    A::deallocate(*bufIter, *sizeIter);
    CUDA_CHECKED_CALL(cudaEventDestroy(*evtIter));

    ++evtIter;
    ++bufIter;
    ++sizeIter;

    evts[bi_omp_tid].pop_front();
    bufs[bi_omp_tid].pop_front();
    sizes[bi_omp_tid].pop_front();

    done = evtIter == evts[bi_omp_tid].end() || cudaEventQuery(*evtIter) != cudaSuccess;
  }
  #endif
}

template<class A>
void bi::pipelined_allocator<A>::add(pointer buf, size_type num) {
  #ifdef ENABLE_CUDA
  if (buf != NULL) {
    /* record event so as to know when buffer can be deallocated */
    cudaEvent_t evt;
    CUDA_CHECKED_CALL(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
    CUDA_CHECKED_CALL(cudaEventRecord(evt));

    evts[bi_omp_tid].push_back(evt);
    bufs[bi_omp_tid].push_back(buf);
    sizes[bi_omp_tid].push_back(num);
  }
  #endif
}

template<class A>
inline void bi::pipelined_allocator<A>::empty() {
  #ifdef ENABLE_CUDA
  BOOST_AUTO(evtIter, evts[bi_omp_tid].begin());
  BOOST_AUTO(bufIter, bufs[bi_omp_tid].begin());
  BOOST_AUTO(sizeIter, sizes[bi_omp_tid].begin());

  while (evtIter != evts[bi_omp_tid].end()) {
    CUDA_CHECKED_CALL(cudaEventSynchronize(*evtIter));
    CUDA_CHECKED_CALL(cudaEventDestroy(*evtIter));

    A::deallocate(*bufIter, *sizeIter);

    ++evtIter;
    ++bufIter;
    ++sizeIter;
  }

  evts[bi_omp_tid].clear();
  bufs[bi_omp_tid].clear();
  sizes[bi_omp_tid].clear();
  #endif
}

#endif
