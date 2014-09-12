/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP

#include "DistributedClientServer.hpp"
#include "mpi.hpp"

namespace bi {
/**
 * Distributed stopper.
 *
 * @ingroup stopper
 *
 * @tparam S Stopper type.
 */
template<class S>
class DistributedStopper: public S, public DistributedClientServer {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(DistributedStopper, S)

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

  /**
   * @copydoc NullHandler::done()
   */
  bool done() const;

  /**
   * @copydoc NullHandler::handleable()
   */
  static bool handleable(const int tag);

  /**
   * @copydoc NullHandler::join()
   */
  void join(MPI_Comm& comm);

  /**
   * @copydoc NullHandler::handle()
   */
  void handle(MPI_Comm& comm, MPI_Status& status);
};
}

template<class S>
bool bi::DistributedStopper<S>::stop(const double maxlw) const {
  boost::optional<boost::mpi::status> status = comm.iprobe(0, MPI_TAG_STOPPER_STOP);
  if (status) {
    comm.recv(status->source(), status->tag());
    return true;
  }
  return false;
}

template<class S>
void bi::DistributedStopper<S>::add(const double lw, const double maxlw) {
  if (parent == NULL) {
    S::add(lw, maxlw);
  } else {
    comm.isend(0, MPI_TAG_STOPPER_ADD_WEIGHT, lw);
    comm.isend(0, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
  }
}

template<class S>
template<class V1>
void bi::DistributedStopper<S>::add(const V1 lws, const double maxlw) {
  if (parent == NULL) {
    S::add(lws, maxlw);
  } else {
    comm.isend(0, MPI_TAG_STOPPER_ADD_WEIGHTS, lws);
    comm.isend(0, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
  }
}

template<class S>
void bi::DistributedStopper<S>::reset() {
  // should really be synchronous...
}

template<class S>
bool bi::DistributedStopper<S>::done() const {
  return stop();
}

template<class S>
bool bi::DistributedStopper<S>::handleable(const int tag) {
  return MPI_TAG_STOPPER_STOP <= tag && tag <= MPI_TAG_STOPPER_MAX_WEIGHT;
}

template<class S>
void bi::DistributedStopper<S>::join(MPI_Comm& comm) {
  //
}

template<class S>
void bi::DistributedStopper<S>::handle(MPI_Comm& comm,
    MPI_Status& status) {
  typedef typename host_temp_vector<real>::type vector_type;

  boost::mpi::communicator comm1(comm, boost::mpi::comm_attach);
  boost::mpi::status status1;
  double maxlw;

  switch (status.tag) {
  case MPI_TAG_STOPPER_ADD_WEIGHTS:
    vector_type lws;
    status1 = comm1.recv(status.source, MPI_TAG_STOPPER_ADD_WEIGHTS, lws);
    status1 = comm1.recv(status.source, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
    add(lws, maxlw);
    break;
  case MPI_TAG_STOPPER_ADD_WEIGHT:
    double lw;
    status1 = comm1.recv(status.source, MPI_TAG_STOPPER_ADD_WEIGHT, lw);
    status1 = comm1.recv(status.source, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
    add(lw, maxlw);
    break;
  default:
    BI_WARN_MSG(false,
        "Unrecognised or out-of-sequence tag from client: " << status.tag);
  }

  if (parent == NULL && stop()) {
    children.ibroadcast();
  }
}

#endif
