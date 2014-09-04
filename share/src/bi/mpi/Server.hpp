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
 *
 * @tparam T Server type.
 */
template<class T>
class Server: public T {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(Server, T)

  /**
   * Open port.
   */
  void open();

  /**
   * Close port.
   */
  void close();

  /**
   * Get port name for client connections.
   */
  const char* getPortName() const;

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
   * Intercommunicators to clients.
   */
  std::set<MPI_Comm> comms;

  /**
   * New intercommunicators ready to be added.
   */
  std::set<MPI_Comm> newcomms;

  /**
   * Old intercommunicators ready to be ejected.
   */
  std::set<MPI_Comm> oldcomms;

  /**
   * Port as written by MPI_Open_port.
   */
  char port_name[MPI_MAX_PORT_NAME];
};
}

template<class T>
void bi::Server<T>::open() {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class T>
void bi::Server<T>::close() {
  err = MPI_Close_port(port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

template<class T>
const char* bi::Server<T>::getPortName() const {
  return port_name;
}

template<class T>
void bi::Server<T>::run() {
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

template<class T>
void bi::Server<T>::accept() {
  int err;
  MPI_Comm comm;
  do {
    err = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
    if (err == MPI_SUCCESS) {
      err = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
      if (err == MPI_SUCCESS) {
        #pragma omp critical
        newcomms.push_back(comm);
      }
    }
  } while (true);
}

template<class T>
void bi::Server<T>::probe() {
  MPI_Status status;
  int flag, err;
  do {
    for (BOOST_AUTO(iter, comms.begin()); iter != comms.end(); ++iter) {
      err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
      if (err == MPI_SUCCESS) {
        if (flag) {
          if (status.tag == MPI_TAG_DISCONNECT) {
            /* graceful disconnect */
            err = MPI_Recv(NULL, 0, MPI_INT, status.source, status.tag, *iter, &status);
            BI_ASSERT(err == MPI_SUCCESS);

            err = MPI_Comm_disconnect(&comm);
            if (err != MPI_SUCCESS) {
              /* ungraceful disconnect */
              err = MPI_Abort(comm, err); ///@todo Does this abort self?
              BI_ASSERT(err == MPI_SUCCESS);
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
        err = MPI_Abort(comm, err); ///@todo Does this abort self?
        BI_ASSERT(err == MPI_SUCCESS);
        oldcomms.insert(comm);
      } else {
        BI_ASSERT(err == MPI_SUCCESS);
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
