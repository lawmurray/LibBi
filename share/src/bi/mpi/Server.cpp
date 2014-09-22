/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Server.hpp"

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

bi::Server::Server(TreeNetworkNode& node) :
    node(node) {
  //
}

const char* bi::Server::getPortName() const {
  return port_name;
}

void bi::Server::open() throw (boost::mpi::exception) {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  if (err != MPI_SUCCESS) {
    boost::throw_exception(boost::mpi::exception("MPI_Open_port", err));
  }
}

void bi::Server::close() throw (boost::mpi::exception) {
  int err = MPI_Close_port(port_name);
  if (err != MPI_SUCCESS) {
    boost::throw_exception(boost::mpi::exception("MPI_Close_port", err));
  }
}

void bi::Server::disconnect(boost::mpi::communicator child,
    boost::mpi::status status) {
  try {
    child.recv(status.source(), status.tag());
    MPI_Comm comm(child);
    int err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      boost::throw_exception(
          boost::mpi::exception("MPI_Comm_disconnect", err));
    }
  } catch (boost::mpi::exception e) {
    //
  }
}
