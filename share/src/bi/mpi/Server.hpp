/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_SERVER_HPP
#define BI_MPI_SERVER_HPP

#include "NullHandler.hpp"
#include "mpi.hpp"

#include <set>

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
 * open() to open a port, getPortName() to recover that port for client
 * processes, and finally run() to run the server.
 */
template<class H1 = NullHandler, class H2 = NullHandler,
    class H3 = NullHandler>
class Server {
public:
  /**
   * Constructor.
   */
  Server();

  /**
   * Get port name for client connections.
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
   * Accept client connections.
   */
  void accept();

  /**
   * Serve client requests.
   */
  void serve();

  /**
   * Join client.
   *
   * @param comm Intercommunicator with the client.
   * @param status Status from probe that led to disconnect.
   */
  void join(MPI_Comm& comm, MPI_Status& status);

  /**
   * Disconnect client.
   *
   * @param comm Intercommunicator with the client.
   * @param status Status from probe that led to disconnect.
   *
   * Used for a bilateral disconnect.
   */
  void disconnect(MPI_Comm& comm, MPI_Status& status);

  /**
   * abort client.
   *
   * @param comm Intercommunicator with the client.
   *
   * Used for a unilateral disconnect. Typically called if the connection
   * with the client fails, or if communication does not adhere to
   * protocol.
   */
  void abort(MPI_Comm& comm);

  /**
   * Handle client message.
   *
   * @param comm Intercommunicator with the client.
   * @param status Status from probe that led to disconnect.
   */
  void handle(MPI_Comm& comm, MPI_Status& status);

  /**
   * Add client.
   *
   * @param comm Communicator associated with the client.
   *
   * @return Number of clients before adding this new client.
   *
   * This queues the client to be added on the next call to updateClients().
   * This method is thread safe.
   */
  int addClient(MPI_Comm& comm);

  /**
   * Remove client.
   *
   * @param comm Communicator associated with the client.
   *
   * This queues the client to be removed on the next call to
   * updateClients(). This method is thread-safe.
   */
  void removeClient(MPI_Comm& comm);

  /**
   * Update client list.
   *
   * @return Number of clients.
   *
   * This adds any This method is thread safe.
   */
  int updateClients();

  /*
   * Handlers.
   */
  H1* h1;
  H2* h2;
  H3* h3;

  /**
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];

  /**
   * Intercommunicators to clients.
   */
  std::set<MPI_Comm> comms;

  /**
   * New intercommunicators ready to be added.
   */
  std::set<MPI_Comm> newcomms;

  /**
   * Old intercommunicators ready to be aborted.
   */
  std::set<MPI_Comm> oldcomms;
};
}

template<class H1, class H2, class H3>
bi::Server<H1,H2,H3>::Server() :
    h1(NULL), h2(NULL), h3(NULL) {
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
   * accept() waiting for client connections, serve() servicing client
   * requests. Rather than starting both now, we start only accept(), which
   * will itself start serve() in a new thread once the first client
   * connects. This avoids a busy-wait in handle() before any clients have
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
            n = queueClient(comm);
            if (n == 0) {
#pragma omp task
              serve();  // start serving clients
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

  while (updateClients() > 0) {
    for (BOOST_AUTO(iter, comms.begin()); iter != comms.end(); ++iter) {
      err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
      if (err == MPI_SUCCESS) {
        if (flag) {
          if (status.tag == MPI_TAG_DISCONNECT) {
            disconnect(*iter, status);
            removeClient (*iter);
          } else {
            handle(*iter, status);
          }
        } else if (err == MPI_ERR_COMM) {
          abort (*iter);
          removeClient(*iter);
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

template<class H1, class H2, class H3>
int bi::Server<H1,H2,H3>::addClient(MPI_Comm& comm) {
  int n = 0;
#pragma omp critical
  {
    n = comms.size() + newcomms.size();
    newcomms.insert(comm);
  }
  return n;
}

template<class H1, class H2, class H3>
void bi::Server<H1,H2,H3>::removeClient(MPI_Comm& comm) {
#pragma omp critical
  {
    oldcomms.insert(comm);
  }
}

template<class H1, class H2, class H3>
int bi::Server<H1,H2,H3>::updateClients() {
  int n = 0;
#pragma omp critical
  {
    comms.insert(newcomms.begin(), newcomms.end());
    newcomms.clear();
    comms.erase(oldcomms.begin(), oldcomms.end());
    oldcomms.clear();
    n = comms.size();

    /* post-conditions */
    BI_ASSERT(newcomms.size() == 0);
    BI_ASSERT(oldcomms.size() == 0);
  }
  return n;
}

#endif
