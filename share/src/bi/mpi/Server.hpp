/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_SERVER_HPP
#define BI_MPI_SERVER_HPP

#include "TreeNetworkNode.hpp"
#include "Handler.hpp"
#include "mpi.hpp"

#include <list>

namespace bi {
/**
 * Server.
 *
 * @ingroup server
 *
 * Set up the server by registering handlers with registerHandler(), then call
 * open() to open a port, getPortName() to recover that port for child
 * processes, and finally run() to run the server.
 */
class Server {
public:
  /**
   * Constructor.
   *
   * @param network Network node.
   */
  Server(TreeNetworkNode& network);

  /**
   * Get port name for child connections.
   */
  const char* getPortName() const;

  /*
   * Set handlers.
   */
  void registerHandler(Handler* handler);

  /**
   * Open port.
   */
  void open();

  /**
   * Close port.
   */
  void close();

  /**
   * Run the server.
   *
   * Does not return until the server terminates.
   */
  void run();

  /**
   * Is all work done?
   */
  bool done() const;

private:
  /**
   * Accept child connections.
   */
  void accept();

  /**
   * Serve child requests.
   */
  void serve();

  /**
   * Join child.
   *
   * @param comm Intercommunicator with the child.
   */
  void join(MPI_Comm comm);

  /**
   * Disconnect child.
   *
   * @param comm Intercommunicator with the child.
   * @param status Status from probe that led to disconnect.
   *
   * Used for a bilateral disconnect.
   */
  void disconnect(MPI_Comm comm, MPI_Status status);

  /**
   * Forcefully disconnect child.
   *
   * @param comm Intercommunicator with the child.
   *
   * Used for a unilateral disconnect. Typically called if the connection
   * with the child fails, or if communication does not adhere to
   * protocol.
   */
  void abort(MPI_Comm comm);

  /**
   * Handle child message.
   *
   * @param comm Intercommunicator with the child.
   * @param status Status from probe that led to disconnect.
   */
  void handle(MPI_Comm comm, MPI_Status status);

  /**
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];

  /**
   * Network node.
   */
  TreeNetworkNode& network;

  /*
   * Handlers.
   */
  std::list<Handler*> handlers;
};
}

#endif
