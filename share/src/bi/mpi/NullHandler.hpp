/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_NULLHANDLER_HPP
#define BI_MPI_NULLHANDLER_HPP

#include "mpi.hpp"

namespace bi {
/**
 * Null handler for client requests of server.
 *
 * @ingroup server
 */
class NullHandler {
public:
  /**
   * Is all work done?
   */
  bool done() const;

  /**
   * Can a message with a given tag be handled?
   *
   * @param tag The tag.
   */
  static bool handleable(const int tag);

  /**
   * Handle a new client joining the computation.
   *
   * @param comm Intercommunicator associated with client.
   */
  void join(MPI_Comm& comm);

  /**
   * Handle a message received from the client.
   *
   * @param comm Intercommunicator associated with client.
   * @param status Status of the probe that found the message.
   *
   * At the very least, this method needs to receive the message found by the
   * probe.
   */
  void handle(MPI_Comm& comm, const MPI_Status& status);
};
}

inline bool bi::NullHandler::done() const {
  return true;
}

inline bool bi::NullHandler::handleable(const int tag) {
  return false;
}

inline void bi::NullHandler::join(MPI_Comm& comm) {
  //
}

inline void bi::NullHandler::handle(MPI_Comm& comm,
    const MPI_Status& status) {
  BI_ASSERT(false);
}

#endif
