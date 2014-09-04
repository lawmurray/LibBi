/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPERSERVER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPERSERVER_HPP

namespace bi {
/**
 * Distributed stopper server.
 *
 * @ingroup stopper
 *
 * @tparam S Stopper type.
 */
template<class S>
class DistributedStopperServer: public S {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(DistributedStopperServer, S)

  /**
   * Service a request.
   */
  void serve(MPI_Comm& comm, MPI_Status& status);
};
}

template<class S>
void bi::DistributedStopperServer<S>::serve(MPI_Comm& comm,
    MPI_Status& status) {
  typedef typename host_temp_vector<real>::type vector_type;

  boost::mpi::communicator comm1(comm, boost::mpi::comm_attach);
  boost::mpi::status status1;

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
}

#endif
