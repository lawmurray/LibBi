/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Client.hpp"

#include "../misc/assert.hpp"

bi::Client::Client(TreeNetworkNode& node) :
    node(node) {
  //
}

void bi::Client::connect(const char* port_name) throw (boost::mpi::exception) {
  MPI_Comm comm;
  int err = MPI_Comm_connect(const_cast<char*>(port_name), MPI_INFO_NULL, 0,
      MPI_COMM_SELF, &comm);
  // ^ MPICH docs suggest first arg should be const char*, but OpenMP, at
  //   least version 1.5.4, uses char* only.
  if (err != MPI_SUCCESS) {
    boost::throw_exception(boost::mpi::exception("MPI_Comm_connect", err));
  }

  err = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
  if (err != MPI_SUCCESS) {
    boost::throw_exception(
        boost::mpi::exception("MPI_Comm_set_errhandler", err));
  }

  boost::mpi::communicator parent(comm, boost::mpi::comm_attach);
  node.parent = parent;
}

void bi::Client::disconnect() {
  try {
    node.parent.isend(0, MPI_TAG_DISCONNECT);
    MPI_Comm comm(node.parent);
    int err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      boost::throw_exception(
          boost::mpi::exception("MPI_Comm_disconnect", err));
    }
  } catch (boost::mpi::exception e) {
    //
  }

  boost::mpi::communicator parent(MPI_COMM_NULL, boost::mpi::comm_attach);
  node.parent = parent;
}
