/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_TREENETWORKNODE_HPP
#define BI_MPI_TREENETWORKNODE_HPP

#include "boost/mpi/communicator.hpp"

#include <forward_list>

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
   */
  std::forward_list<boost::mpi::communicator> children;
};
}

#endif
