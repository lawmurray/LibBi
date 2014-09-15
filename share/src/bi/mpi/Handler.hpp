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

#include "boost/mpi/communicator.hpp"

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
   * Handle a new child joining the computation.
   *
   * @param child The child.
   */
  virtual void join(boost::mpi::communicator child);

  /**
   * Can a message with a given tag be handled?
   *
   * @param tag The tag.
   */
  virtual bool canHandle(const int tag) const;

  /**
   * Handle a message received from a child.
   *
   * @param child The child.
   * @param status Status of the probe that found the message.
   *
   * At the very least, this method needs to receive the message found by the
   * probe.
   */
  virtual void handle(boost::mpi::communicator child,
      boost::mpi::status status);
};
}

#endif
