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

#include <set>

namespace bi {
/**
 * Server wrapper, buckles a common interface onto any server.
 *
 * @ingroup server
 */
template<class S>
class Server: public S {
public:
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
   * Listen for connections.
   */
  void accept();

  /**
   * Probe for messages.
   */
  void probe();

  /**
   * Communicators, one for each connected client.
   */
  std::set<MPI_Comm> comms;

  /**
   * New communicators ready to be added.
   */
  std::set<MPI_Comm> newcomms;

  /**
   * Old communicators ready to be ejected.
   */
  std::set<MPI_Comm> oldcomms;

  /**
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];
};
}

template<class S>
void bi::Server<S>::open() {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class S>
void bi::Server<S>::close() {
  err = MPI_Close_port(port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class S>
void bi::Server<S>::run() {
  #ifndef ENABLE_OPENMP
  BI_ERROR_MSG(false, "A server requires OpenMP to run, use --enable-openmp.")
  #endif

  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task
      accept();

      #pragma omp task
      probe();
    }
  }
}

template<class S>
void bi::Server<S>::accept() {
  int err;
  MPI_Comm comm;
  do {
    err = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
    if (err == MPI_SUCCESS) {
      #pragma omp critical
      newcomms.push_back(comm);
    }
  } while (true);
}

template<class S>
void bi::Server<S>::probe() {
  MPI_Status status;
  int flag, err;
  do {
    for (BOOST_AUTO(iter, comms.begin()); iter != comms.end(); ++iter) {
      err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);

      if (err == MPI_SUCCESS) {
        if (flag) {
          if (status.tag == TAG_DISCONNECT) {
            /* graceful disconnect */
            err = MPI_Comm_disconnect(&comm);
            if (err != MPI_SUCCESS) {
              /* ungraceful disconnect */
              MPI_Abort(comm, err); ///@todo Does this abort self?
            }
            oldcomms.insert(comm);
          } else {
            /* service request */
            #pragma omp task
            this->serve(*iter, status);
          }
        }
      } else if (err == MPI_ERR_COMM) {
        /* ungraceful disconnect */
        MPI_Abort(comm, err); ///@todo Does this abort self?
        oldcomms.insert(comm);
      }
    }

    /* eject old communicators */
    comms.erase(oldcomms.begin(), oldcomms.end());
    oldcomms.clear();

    /* insert new communicators */
    #pragma omp critical
    {
      comms.insert(newcomms.begin(), newcomms.end());
      newcomms.clear();
    }
  } while (!(this->stop() && comms.empty()));
}

#endif
