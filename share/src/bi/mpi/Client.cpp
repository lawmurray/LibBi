/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Client.hpp"

bi::Client::Client(TreeNetworkNode& network) : network(network) {
  //
}

void bi::Client::connect(const char* port_name) {
  MPI_Comm comm;
  err = MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
  BI_ERROR_MSG(err == MPI_SUCCESS, "Could not connect to server");
  network.setParent(comm);
}

void bi::Client::disconnect() {
  int err;
  MPI_Request request;
  MPI_Comm comm = network.getParent();

  err = MPI_Isend(NULL, 0, MPI_INT, 0, MPI_TAG_DISCONNECT, comm, &request);
  if (err != MPI_SUCCESS) {
    err = MPI_Abort(comm, err);
  } else {
    err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      err = MPI_Abort(comm, err);
    }
    network.setParent(MPI_COMM_NULL);
  }
}
