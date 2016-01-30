/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_CLIENT_HPP
#define BI_MPI_CLIENT_HPP

#include "TreeNetworkNode.hpp"
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
   * Constructor.
   *
   * @param node Network node.
   */
  Client(TreeNetworkNode& node);

  /**
   * Connect to server.
   *
   * @param port_name Port name of server, as returned by MPI_Open_port() on
   * that server.
   */
  void connect(const char* port_name) throw (boost::mpi::exception);

  /**
   * Disconnect from server.
   */
  void disconnect();

private:
  /**
   * Network node.
   */
  TreeNetworkNode& node;
};
}

#endif
