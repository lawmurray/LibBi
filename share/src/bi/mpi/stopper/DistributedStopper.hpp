/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP

#include "Handler.hpp"
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
class DistributedStopper: public S, public Handler {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(DistributedStopper, S)

  /**
   * Destructor.
   */
  virtual ~DistributedStopper();

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

  /*
   * Inherited from Handler.
   */
  virtual bool done() const;
  virtual bool canHandle(const int tag) const;
  virtual void join(MPI_Comm comm);
  virtual void handle(MPI_Comm comm, MPI_Status status);

private:
  /**
   * Send current buffer up to parent.
   */
  void sendUp();

  /**
   * Request handle for sends to parent.
   */
  boost::mpi::request request;
};
}

template<class S>
bi::DistributedStopper<S>::~DistributedStopper() {
  /* finish outstanding send */
  request.wait();

  /* send any remaining */
  sendUp();
  request.wait();
}

template<class S>
bool bi::DistributedStopper<S>::stop(const double maxlw) const {
  boost::optional<boost::mpi::status> status = network.getParent().iprobe(0, MPI_TAG_STOPPER_STOP);
  if (status) {
    comm.recv(status->source(), status->tag());
  }
  return status;
}

template<class S>
void bi::DistributedStopper<S>::add(const double lw, const double maxlw) {
  S::add(lw, maxlw);
  sendUp();
}

template<class S>
template<class V1>
void bi::DistributedStopper<S>::add(const V1 lws, const double maxlw) {
  S::add(lws, maxlw);
  sendUp();
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
bool bi::DistributedStopper<S>::canHandle(const int tag) const {
  return MPI_TAG_STOPPER_STOP <= tag && tag <= MPI_TAG_STOPPER_MAX_WEIGHT;
}

template<class S>
void bi::DistributedStopper<S>::join(MPI_Comm comm) {
  //
}

template<class S>
void bi::DistributedStopper<S>::handle(MPI_Comm comm, MPI_Status status) {
  typedef typename host_temp_vector<real>::type vector_type;

  boost::mpi::communicator comm1(comm, boost::mpi::comm_attach);
  double maxlw = 0.0;

  switch (status.tag) {
  case MPI_TAG_STOPPER_ADD_WEIGHTS:
    vector_type lws;
    comm1.recv(status.source(), MPI_TAG_STOPPER_ADD_WEIGHTS, lws);
    add(lws, maxlw);
    break;
  case MPI_TAG_STOPPER_ADD_WEIGHT:
    double lw;
    comm1.recv(status.source(), MPI_TAG_STOPPER_ADD_WEIGHT, lw);
    add(lw, maxlw);
    break;
  default:
    BI_WARN_MSG(false,
        "Unrecognised or out-of-sequence tag from client: " << status.tag);
  }

  if (network.hasParent() && stop()) {
    sendDown();
  }
}

template<class S>
void bi::DistributedStopper<S>::sendUp() {
  bool flag = true;
  if (full()) {
    request.wait();
  } else {
    flag = request.test();
  }
  if (flag) {
    request = network.getParent().isend(X, 0, MPI_TAG_STOPPER_ADD_WEIGHT);
  }
}

template<class S>
void bi::DistributedStopper<S>::sendDown() {
  std::list<boost::mpi::request> requests;
  for (BOOST_AUTO(iter, network.getChildren().begin()); iter != network.getChildren().end(); ++iter) {
    requests.push_back(iter->isend(MPI_TAG_STOPPER_STOP));
  }
  boost::mpi::wait_all(requests.begin(); requests.end());
}

#endif
