/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_CLIENT_HPP
#define BI_MPI_CLIENT_HPP

#include "mpi.hpp"
#include "../misc/macro.hpp"

namespace bi {
/**
 * Client wrapper, buckles a common interface onto any client.
 *
 * @ingroup server
 *
 * @tparam T Client type.
 */
template<class T>
class Client: public T {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(Client, T)

  /**
   * Connect to server.
   *
   * @param port_name Port name of server, as returned by MPI_Open_port() on
   * that server.
   */
  void connect(const char* port_name);

  /**
   * Disconnect from server.
   */
  void disconnect();

private:
  /**
   * Intercommunicator to server.
   */
  MPI_Comm comm;
};
}

template<class T>
void bi::Client<T>::connect(const char* port_name) {
  err = MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class T>
void bi::Client<T>::disconnect() {
  int err;
  MPI_Request request;

  err = MPI_Isend(NULL, 0, MPI_INT, 0, MPI_TAG_DISCONNECT, comm, &request);
  if (err == MPI_SUCCESS) {
    err = MPI_Comm_disconnect(&comm);
    BI_ASSERT(err == MPI_SUCCESS);
  } else if (err == MPI_COMM_ERR) {
    err = MPI_Abort(comm, err);
    BI_ASSERT(err == MPI_SUCCESS); ///@todo Does this abort self?
  }
}

#endif
