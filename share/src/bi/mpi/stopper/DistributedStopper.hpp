/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP

#include "../TreeNetworkNode.hpp"
#include "../mpi.hpp"

namespace bi {
/**
 * Distributed stopper.
 *
 * @ingroup stopper
 *
 * @tparam S Stopper type.
 */
template<class S>
class DistributedStopper {
public:
  /**
   * Constructor.
   *
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
  bool stop(const double maxlw) const;

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw);

  /**
   * @copydoc Stopper::add(const V1, const double)
   */
  template<class V1>
  void add(const V1 lws, const double maxlw);

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
   * Base (local) stopper.
   */
  S& base;

  /**
   * Network node.
   */
  TreeNetworkNode& node;

  /**
   * Request handle for sends to parent.
   */
  boost::mpi::request request;

  /**
   * Stop?
   */
  bool stop;
};
}

template<class S>
bi::DistributedStopper<S>::DistributedStopper(S& base, TreeNetworkNode& node) :
    base(base), node(node), stop(false) {
  //
}

template<class S>
bi::DistributedStopper<S>::~DistributedStopper() {
  finish();
}

template<class S>
bool bi::DistributedStopper<S>::stop(const double maxlw) const {
  if (!stop) {
    boost::optional < boost::mpi::status > status = node.parent.iprobe(0,
        MPI_TAG_STOPPER_STOP);
    if (status) {
      comm.recv(status->source(), status->tag());
      stop = true;
    }
  }
  return stop;
}

template<class S>
void bi::DistributedStopper<S>::add(const double lw, const double maxlw) {
  base.add(lw, maxlw);
  send();
}

template<class S>
template<class V1>
void bi::DistributedStopper<S>::add(const V1 lws, const double maxlw) {
  base.add(lws, maxlw);
  send();
}

template<class S>
void bi::DistributedStopper<S>::reset() {
  finish();
  base.reset();
  stop = false;
}

template<class S>
bool bi::DistributedStopper<S>::done() const {
  return stop();
}

template<class S>
void bi::DistributedStopper<S>::send() {
  if (node.parent != MPI_COMM_NULL) {
    bool flag = true;
    if (full()) {
      request.wait();
    } else {
      flag = request.test();
    }
    if (flag) {
      request = node.parent.isend(X, 0, MPI_TAG_STOPPER_ADD_WEIGHT);
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
