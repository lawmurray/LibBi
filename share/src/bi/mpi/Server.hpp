/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_SERVER_HPP
#define BI_MPI_SERVER_HPP

#include "mpi.hpp"
#include "TreeNetworkNode.hpp"

#include "boost/typeof/typeof.hpp"

namespace bi {
/**
 * Server.
 *
 * @ingroup server
 *
 * Call open() to open a port, getPortName() to recover that port for child
 * processes, and finally run() to run the server, giving an appropriate
 * handler for incoming messages.
 */
class Server {
public:
  /**
   * Constructor.
   *
   * @param node Network node.
   */
  Server(TreeNetworkNode& node);

  /**
   * Get port name for child connections.
   */
  const char* getPortName() const;

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
   * @tparam H Handler type.
   *
   * @param Handler for messages received.
   *
   * Does not return until the server terminates.
   */
  template<class H>
  void run(H& handler);

private:
  /**
   * Accept child connections.
   *
   * @tparam H Handler type.
   *
   * @param Handler for messages received.
   */
  template<class H>
  void accept(H& handler);

  /**
   * Serve child requests.
   *
   * @tparam H Handler type.
   *
   * @param Handler for messages received.
   */
  template<class H>
  void serve(H& handler);

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
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];

  /**
   * Network node.
   */
  TreeNetworkNode& node;
};
}

template<class H>
void bi::Server::run(H& handler) {
  /*
   * The methods accept() and serve() are designed to run concurrently,
   * accept() waiting for child connections, serve() servicing child
   * requests. Rather than starting both now, we start only accept(), which
   * will itself start serve() in a new thread once the first child
   * connects. This avoids a busy-wait in handle() before any children have
   * connected. It does not avoid a busy-wait in accept() if that is how the
   * particular MPI implementation implements MPI_Comm_accept(), but we can't
   * do anything about that.
   */
  accept(handler);
}

template<class H>
void bi::Server::accept(H& handler) {
  int err, n;
  MPI_Comm comm;
#pragma omp parallel
  {
#pragma omp single
    {
      do {
        try {
          err = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF,
              &comm);
          if (err != MPI_SUCCESS) {
            boost::throw_exception(
                boost::mpi::exception("MPI_Comm_accept", err));
          }

          err = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
          if (err != MPI_SUCCESS) {
            boost::throw_exception(
                boost::mpi::exception("MPI_Comm_set_errhandler", err));
          }

          boost::mpi::communicator child(comm, boost::mpi::comm_attach);
          handler.init(child);
          BOOST_AUTO(iter, node.children.push_front(child));
          if (++iter == node.children.end()) {
#pragma omp task
            serve(handler);  // start serving children
          }
        } catch (boost::mpi::exception e) {
          //
        }
      } while (!handler.done());
    }
  }
}

template<class H>
void bi::Server::serve(H& handler) {
  MPI_Status status;
  int flag, err;

  /* service messages */
  while (!node.children.empty()) {
    BOOST_AUTO(prev, node.children.before_begin());
    BOOST_AUTO(iter, node.children.begin());
    for (; iter != node.children.end(); prev = iter++) {
      try {
        err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
        /* use MPI_Iprobe and not iter->iprobe, as latter can't distinguish
         * between error and no message */
        ///@todo Use MPI_Improbe for multithreaded handler
        if (err != MPI_SUCCESS) {
          boost::throw_exception(boost::mpi::exception("MPI_Iprobe", err));
        }
        if (flag) {
          if (status.MPI_TAG == MPI_TAG_DISCONNECT) {
            disconnect(*iter, status);
            node.children.erase_after(prev);
          } else {
            handler.handle(*iter, status);
          }
        }
      } catch (boost::mpi::exception e) {
        node.children.erase_after(prev);
      }
    }

    /* clean up outstanding requests */
    BOOST_AUTO(prevRequests, node.requests.before_begin());
    BOOST_AUTO(iterRequests, node.requests.begin());
    for (; iterRequests != node.requests.end(); prevRequests = iterRequests++) {
      try {
        if (iterRequests->test()) {
          node.requests.erase_after(prevRequests);
        }
      } catch (boost::mpi::exception e) {
        node.requests.erase_after(prevRequests);
      }
    }
  }
}

#endif
