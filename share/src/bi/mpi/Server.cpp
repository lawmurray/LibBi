/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Server.hpp"

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

bi::Server::Server(TreeNetworkNode& network) :
    network(network) {
  //
}

const char* bi::Server::getPortName() const {
  return port_name;
}

void bi::Server::registerHandler(Handler* handler) {
  handlers.push_back(handler);
}

void bi::Server::open() {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

void bi::Server::close() {
  int err = MPI_Close_port(port_name);
  BI_ERROR(err == MPI_SUCCESS);
}

void bi::Server::run() {
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

bool bi::Server::done() const {
  bool done = true;
  for (BOOST_AUTO(iter, handlers.begin()); iter != handlers.end(); ++iter) {
    BOOST_AUTO(handler, *iter);
    done = done && handler->done();
  }
  return done;
}

void bi::Server::accept() {
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
      } while (!done());
    }
  }
}

void bi::Server::serve() {
  MPI_Status status;
  int flag, err;

  while (network.updateChildren() > 0) {
    for (BOOST_AUTO(iter, network.getChildren().begin());
        iter != network.getChildren().end(); ++iter) {
      err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
      if (err == MPI_SUCCESS) {
        if (flag) {
          if (status.MPI_TAG == MPI_TAG_DISCONNECT) {
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

void bi::Server::join(MPI_Comm comm) {
  for (BOOST_AUTO(iter, handlers.begin()); iter != handlers.end(); ++iter) {
    BOOST_AUTO(handler, *iter);
    handler->join(comm);
  }
}

void bi::Server::disconnect(MPI_Comm comm, MPI_Status status) {
  int err = MPI_Recv(NULL, 0, MPI_INT, status.MPI_SOURCE, status.MPI_TAG,
      comm, NULL);
  if (err != MPI_SUCCESS) {
    abort(comm);
  } else {
    err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      abort(comm);
    }
  }
}

void bi::Server::abort(MPI_Comm comm) {
  int err = MPI_Abort(comm, err);
  BI_ASSERT(err == MPI_SUCCESS);
}

void bi::Server::handle(MPI_Comm comm, MPI_Status status) {
  const unsigned tag = status.MPI_TAG;
  for (BOOST_AUTO(iter, handlers.begin()); iter != handlers.end(); ++iter) {
    BOOST_AUTO(handler, *iter);
    if (handler->canHandle(tag)) {
      handler->handle(comm, status);
      return;
    }
  }

  /* client is misbehaving */
  abort(comm);
}
