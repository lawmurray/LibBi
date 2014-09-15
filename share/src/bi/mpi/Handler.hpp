/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_HANDLER_HPP
#define BI_MPI_HANDLER_HPP

#include "mpi.hpp"

namespace bi {
/**
 * Handler for client requests of server.
 *
 * @ingroup server
 */
class Handler {
public:
  /**
   * Destructor.
   */
  virtual ~Handler();

  /**
   * Is all work done?
   */
  virtual bool done() const;

  /**
   * Handle a new client joining the computation.
   *
   * @param comm Intercommunicator associated with client.
   */
  virtual void join(MPI_Comm comm);

  /**
   * Can a message with a given tag be handled?
   *
   * @param tag The tag.
   */
  virtual bool canHandle(const int tag) const;

  /**
   * Handle a message received from the client.
   *
   * @param comm Intercommunicator associated with client.
   * @param status Status of the probe that found the message.
   *
   * At the very least, this method needs to receive the message found by the
   * probe.
   */
  virtual void handle(MPI_Comm comm, MPI_Status status);
};
}

#endif
