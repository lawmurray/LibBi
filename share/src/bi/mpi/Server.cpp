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

bi::Server::Server(TreeNetworkNode& node) :
    node(node) {
  //
}

const char* bi::Server::getPortName() const {
  return port_name;
}

void bi::Server::registerHandler(Handler* handler) {
  handlers.push_back(handler);
}

void bi::Server::open() throw (boost::mpi::exception) {
  int err = MPI_Open_port(MPI_INFO_NULL, port_name);
  if (err != MPI_SUCCESS) {
    boost::throw_exception(boost::mpi::exception("MPI_Open_port", err));
  }
}

void bi::Server::close() throw (boost::mpi::exception) {
  int err = MPI_Close_port(port_name);
  if (err != MPI_SUCCESS) {
    boost::throw_exception(boost::mpi::exception("MPI_Close_port", err));
  }
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
          join(child);
          n = node.children.push_front(child);
          if (n == 0) {
#pragma omp task
            serve();  // start serving children
          }
        } catch (boost::mpi::exception e) {
          //
        }
      } while (!done());
    }
  }
}

void bi::Server::serve() {
  MPI_Status status;
  int flag, err;

  while (!node.children.empty()) {
    BOOST_AUTO(iter, node.children.begin());
    BOOST_AUTO(prev, node.childre.before_begin());
    for (; iter != node.children.end(); prev = iter++) {
      try {
        err = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *iter, &flag, &status);
        /* use MPI_Iprobe and not iter->iprobe, as latter can't distinguish
         * between error and no message */
        if (err != MPI_SUCCESS) {
          boost::throw_exception(boost::mpi::exception("MPI_Iprobe", err));
        }
        if (flag) {
          if (status.MPI_TAG == MPI_TAG_DISCONNECT) {
            disconnect(*iter, status);
            node.children.erase_after(prev);
          } else {
            handle(*iter, status);
          }
        }
      } catch (boost::mpi::exception e) {
        node.erase_after(prev);
      }
    }
  }
}

void bi::Server::join(boost::mpi::communicator child) {
  for (BOOST_AUTO(iter, handlers.begin()); iter != handlers.end(); ++iter) {
    BOOST_AUTO(handler, *iter);
    handler->join(child);
  }
}

void bi::Server::disconnect(boost::mpi::communicator child,
    boost::mpi::status status) {
  try {
    child.recv(status.source(), status.tag());
    MPI_Comm comm(child);
    int err = MPI_Comm_disconnect(&comm);
    if (err != MPI_SUCCESS) {
      boost::throw_exception(
          boost::mpi::exception("MPI_Comm_disconnect", err));
    }
  } catch (boost::mpi::exception e) {
    //
  }
}

void bi::Server::handle(boost::mpi::communicator child,
    boost::mpi::status status) {
  const unsigned tag = status.tag();
  for (BOOST_AUTO(iter, handlers.begin()); iter != handlers.end(); ++iter) {
    BOOST_AUTO(handler, *iter);
    if (handler->canHandle(tag)) {
      handler->handle(child, status);
      return;
    }
  }

  /* child is misbehaving */
  BI_WARN_MSG(false, "child misbehaving");
}
