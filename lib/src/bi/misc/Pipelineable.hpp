/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_PIPELINEABLE_HPP
#define BI_MISC_PIPELINEABLE_HPP

#include <list>

namespace bi {
/**
 * Utility base for pipelining copies from host to device.
 *
 * @ingroup misc
 *
 * @tparam T Buffer type.
 *
 * Utility base for operations on the host that must copy their
 * results to device. Each time the operation is performed, results are
 * written to a new temporary host buffer and copied from this to device.
 * Rather than synchronising with and deleting the host buffer immediately,
 * or synchronising to ensure safe reuse of the buffer, a call to add()
 * records an event to allow checking when the copy is complete. The clean()
 * function is used at any time in future to check the status of all
 * recorded copies, and release host buffers pertaining to those which have
 * completed.
 *
 * Typical use of this class is for disk I/O operations, which must write to
 * host buffers as an intermediate, even if required for device operations.
 */
template<class T>
class Pipelineable {
public:
  /**
   * Add buffer.
   */
  void add(T* buf);

  /**
   * Clean up. Checks status of copies and releases buffers where done.
   */
  void clean();

private:
  /**
   * Host-side buffers.
   */
  std::list<T*> bufs;

  /**
   * Events to track host to device copies.
   */
  std::list<cudaEvent_t> evts;
};
}

#include "boost/typeof/typeof.hpp"

template<class T>
void bi::Pipelineable<T>::add(T* buf) {
  /* record event so we can check when copy is complete */
  cudaEvent_t evt;
  CUDA_CHECKED_CALL(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
  CUDA_CHECKED_CALL(cudaEventRecord(evt));

  bufs.push_back(buf);
  evts.push_back(evt);

  /* post-condition */
  assert (bufs.size() == evts.size());
}

template<class T>
void bi::Pipelineable<T>::clean() {
  BOOST_AUTO(evtIter, evts.begin());
  BOOST_AUTO(bufIter, bufs.begin());
  while (evtIter != evts.end()) {
    if (cudaEventQuery(*evtIter) == cudaSuccess) {
      /* copy is done, release */
      CUDA_CHECKED_CALL(cudaEventDestroy(*evtIter));
      delete *bufIter;

      BOOST_AUTO(evtDel, evtIter);
      BOOST_AUTO(bufDel, bufIter);

      ++evtIter;
      ++bufIter;

      evts.erase(evtDel);
      bufs.erase(bufDel);
    }
  }

  /* post-condition */
  assert (bufs.size() == evts.size());
}

#endif
