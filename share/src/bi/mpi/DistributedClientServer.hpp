/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_DISTRIBUTEDCLIENTSERVER_HPP
#define BI_MPI_DISTRIBUTEDCLIENTSERVER_HPP

#include "mpi.hpp"

namespace bi {
/**
 * Base class for client-server hierarchies.
 *
 * @ingroup server
 */
class DistributedClientServer {
public:
  /**
   * Constructor.
   */
  DistributedClientServer();

  /**
   * Set client.
   */
  void setClient(Client* client);

  /**
   * Set server.
   */
  void setServer(Server* server);

protected:
  /**
   * Client.
   */
  Client* client;

  /**
   * Server.
   */
  Server* server;
};
}

#endif
