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
#include "NullHandler.hpp"
#include "mpi.hpp"

namespace bi {
/**
 * Server.
 *
 * @ingroup server
 *
 * @tparam H1 Handler type.
 * @tparam H2 Handler type.
 * @tparam H3 Handler type.
 *
 * Set up the server by registering handlers with setHandler(), then call
 * open() to open a port, getPortName() to recover that port for child
 * processes, and finally run() to run the server.
 */
template<class H1 = NullHandler, class H2 = NullHandler,
    class H3 = NullHandler>
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
  void setHandler(H1* h1);
  void setHandler(H2* h2);
  void setHandler(H3* h3);

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
   * @param status Status from probe that led to disconnect.
   */
  void join(MPI_Comm& comm, MPI_Status& status);

  /**
   * Disconnect child.
   *
   * @param comm Intercommunicator with the child.
   * @param status Status from probe that led to disconnect.
   *
   * Used for a bilateral disconnect.
   */
  void disconnect(MPI_Comm& comm, MPI_Status& status);

  /**
   * Forcefully disconnect child.
   *
   * @param comm Intercommunicator with the child.
   *
   * Used for a unilateral disconnect. Typically called if the connection
   * with the child fails, or if communication does not adhere to
   * protocol.
   */
  void abort(MPI_Comm& comm);

  /**
   * Handle child message.
   *
   * @param comm Intercommunicator with the child.
   * @param status Status from probe that led to disconnect.
   */
  void handle(MPI_Comm& comm, MPI_Status& status);

  /**
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];

  /*
   * Handlers.
   */
  H1* h1;
  H2* h2;
  H3* h3;

  /**
   * Network node.
   */
  TreeNetworkNode& network;
};
}

template<class H1, class H2, class H3>
bi::Server<H1,H2,H3>::Server(TreeNetworkNode& network) :
    h1(NULL), h2(NULL), h3(NULL), network(network) {
  //
}

template<class H1, class H2, class H3>
const char* bi::Server<H1,H2,H3>::getPortName() const {
  return port_name;
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::setHandler(H1* h1) {
  this->h1 = h1;
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::setHandler(H2* h2) {
  this->h2 = h2;
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::setHandler(H3* h3) {
  this->h3 = h3;
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::open() {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::close() {
  err = MPI_Close_port(port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::run() {
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
  accept();
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::accept() {
  int err, n;
  MPI_Comm comm;
#pragma omp parallel
  {
#pragma omp single
    {
      do {
        err = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF,
            &comm);
        if (err == MPI_SUCCESS) {
          err = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
          if (err != MPI_SUCCESS) {
            abort(comm);
          } else {
            join(comm);
            n = network.addChild(comm);
            if (n == 0) {
#pragma omp task
              serve();  // start serving children
            }
          }
        }
      } while (!h1->done() || !h2->done() || !h3->done());
    }
  }
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::serve() {
  MPI_Comm comm;
  MPI_Status status;
  int flag, err;

  while (network.updateChildren() > 0) {
    for (BOOST_AUTO(iter, network.children().begin());
        iter != network.children().end(); ++iter) {
      err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
      if (err == MPI_SUCCESS) {
        if (flag) {
          if (status.tag == MPI_TAG_DISCONNECT) {
            disconnect(*iter, status);
            network.removeChild(*iter);
          } else {
            handle(*iter, status);
          }
        } else if (err == MPI_ERR_COMM) {
          abort (*iter);
          network.removeChild(*iter);
        } else {
          BI_ASSERT(err == MPI_SUCCESS);
        }
      }
    }
  }
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::join(MPI_Comm& comm, MPI_Status& status) {
  int err = MPI_Recv(NULL, 0, MPI_INT, status.source, status.tag, comm, NULL);
  if (err != MPI_SUCCESS) {
    abort(comm);
  } else {
    if (h1 != NULL) {
      h1->join(comm);
    }
    if (h2 != NULL) {
      h2->join(comm);
    }
    if (h3 != NULL) {
      h3->join(comm);
    }
  }
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::disconnect(MPI_Comm& comm, MPI_Status& status) {
  int err = MPI_Recv(NULL, 0, MPI_INT, status.source, status.tag, comm, NULL);
  if (err != MPI_SUCCESS) {
    abort(comm);
  } else {
    err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      abort(comm);
    }
  }
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::abort(MPI_Comm& comm) {
  int err = MPI_Abort(comm, err);
  BI_ASSERT(err == MPI_SUCCESS);
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::handle(MPI_Comm& comm, MPI_Status& status) {
  if (h1 != NULL && H1::handleable(status.tag)) {
    h1->handle(*iter, status);
  } else if (h2 != NULL && H2::handleable(status.tag)) {
    h2->handle(*iter, status);
  } else if (h3 != NULL && H3::handleable(status.tag)) {
    h3->handle(*iter, status);
  } else {
    abort (*iter);
  }
}

#endif
