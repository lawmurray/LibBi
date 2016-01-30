/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_TREENETWORKNODE_HPP
#define BI_MPI_TREENETWORKNODE_HPP

#include "mpi.hpp"
#include "../primitive/forward_list.hpp"

namespace bi {
/**
 * Node of a tree-structure network.
 *
 * @ingroup server
 */
struct TreeNetworkNode {
  /**
   * Constructor.
   */
  TreeNetworkNode();

  /**
   * Intercommunicator to parent.
   */
  boost::mpi::communicator parent;

  /**
   * Intercommunicators to children.
   *
   * @todo Thread-safe forward list implementation.
   */
  forward_list<boost::mpi::communicator> children;

  /**
   * Outstanding requests.
   *
   * @todo Thread-safe forward list implementation.
   */
  forward_list<boost::mpi::request> requests;
};
}

#endif
