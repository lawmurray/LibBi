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

namespace bi {
/**
 * Client.
 *
 * @ingroup server
 */
class Client {
public:
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

#endif
