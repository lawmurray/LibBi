/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP

#include "../mpi.hpp"
#include "../TreeNetworkNode.hpp"
#include "../../cache/Cache1D.hpp"

namespace bi {
/**
 * Distributed stopper.
 *
 * @ingroup stopper
 *
 * @tparam S Stopper type.
 *
 * DistributedStopper is designed to work with a client-server architecture.
 * The generic implementation merely passes weights added in a client process
 * onto the server process. This generic approach works in all cases. There
 * is scope to explicitly implement specialisations of the class template for
 * particular stopper types in order to perform some share of aggregation on
 * the client to reduce message sizes.
 */
template<class S>
class DistributedStopper {
public:
  /**
   * Constructor.
   *
   * @param base Base stopper.
   * @param node Network node.
   */
  DistributedStopper(S& base, TreeNetworkNode& node);

  /**
   * Destructor.
   */
  ~DistributedStopper();

  /**
   * @copydoc Stopper::stop()
   */
  bool stop(const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::add(const V1, const double)
   */
  template<class V1>
  void add(const V1 lws, const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::reset()
   */
  void reset();

private:
  /**
   * Send buffer up to parent.
   */
  void send();

  /**
   * Finish sends.
   */
  void finish();

  /**
   * Network node.
   */
  TreeNetworkNode& node;

  /**
   * Request handle for sends to parent.
   */
  boost::mpi::request request;

  /**
   * Cache of log-weights currently being sent.
   */
  Cache1D<real> cacheSend;

  /**
   * Cache of log-weights currently being accumulated.
   */
  Cache1D<real> cacheAccum;

  /**
   * Index into cacheSend for next write.
   */
  int pSend;

  /**
   * Index into cacheAccum for next write.
   */
  int pAccum;

  /**
   * Stop?
   */
  bool flagStop;

  /**
   * Maximum number of accumulated weights before blocking.
   */
  static const int MAX_ACCUM = 1024;
};
}

template<class S>
bi::DistributedStopper<S>::DistributedStopper(S& base, TreeNetworkNode& node) :
    node(node), pSend(0), pAccum(0), flagStop(false) {
  //
}

template<class S>
bi::DistributedStopper<S>::~DistributedStopper() {
  finish();
}

template<class S>
bool bi::DistributedStopper<S>::stop(const double maxlw) {
  if (!flagStop) {
    boost::optional < boost::mpi::status > status = node.parent.iprobe(0,
        MPI_TAG_STOPPER_STOP);
    if (status) {
      node.parent.recv(status->source(), status->tag());
      flagStop = true;
    }
  }
  return flagStop;
}

template<class S>
void bi::DistributedStopper<S>::add(const double lw, const double maxlw) {
  cacheAccum.set(pAccum++, lw);
  send();
}

template<class S>
template<class V1>
void bi::DistributedStopper<S>::add(const V1 lws, const double maxlw) {
  cacheAccum.set(pAccum++, lws.size(), lws);
  send();
}

template<class S>
void bi::DistributedStopper<S>::reset() {
  finish();
  cacheSend.clear();
  cacheAccum.clear();
  pSend = 0;
  pAccum = 0;
  flagStop = false;
}

template<class S>
void bi::DistributedStopper<S>::send() {
  if (node.parent != MPI_COMM_NULL) {
    bool flag = true;
    if (pAccum == MAX_ACCUM) {
      request.wait();
    } else {
      flag = request.test();
    }
    if (flag) {
      cacheSend.swap(cacheAccum);
      cacheAccum.clear();
      pSend = pAccum;
      pAccum = 0;

      request = node.parent.isend(0, MPI_TAG_STOPPER_LOGWEIGHTS,
          cacheSend.get(0, pSend).buf(), pSend);
    }
  }
}

template<class S>
void bi::DistributedStopper<S>::finish() {
  /* finish outstanding send */
  request.wait();

  /* send any remaining */
  send();
  request.wait();
}

#endif
