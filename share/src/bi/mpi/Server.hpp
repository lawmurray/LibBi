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

#include "boost/mpi/communicator.hpp"

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
  void open() throw (boost::mpi::exception);

  /**
   * Close port.
   */
  void close() throw (boost::mpi::exception);

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
   * @param comm The child.
   */
  void join(const boost::mpi::communicator child);

  /**
   * Disconnect child.
   *
   * @param child The child.
   * @param status Status from probe that led to disconnect.
   *
   * Used for a bilateral disconnect.
   */
  void disconnect(boost::mpi::communicator child, boost::mpi::status status);

  /**
   * Handle child message.
   *
   * @param child The child.
   * @param status Status from probe that led to disconnect.
   */
  void handle(boost::mpi::communicator child, boost::mpi::status status);

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
