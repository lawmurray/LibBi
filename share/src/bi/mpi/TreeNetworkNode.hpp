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

#include <set>

namespace bi {
/**
 * TreeNetworkNode.
 *
 * @ingroup server
 */
class TreeNetworkNode {
public:
  /**
   * Constructor.
   */
  TreeNetworkNode();

  /**
   * Get parent.
   */
  boost::mpi::communicator getParent() const;

  /**
   * Set parent.
   */
  void setParent(boost::mpi::communicator comm);

  /**
   * Get children.
   */
  std::set<boost::mpi::communicator> getChildren();

  /**
   * Add child.
   *
   * @param comm Communicator associated with the child.
   *
   * @return Number of children before adding this new child.
   */
  int addChild(boost::mpi::communicator comm);

  /**
   * Remove child.
   *
   * @param comm Communicator associated with the child.
   *
   * @return Number of children after removing this child.
   */
  int removeChild(boost::mpi::communicator comm);

private:
  /**
   * Communicator to parent.
   */
  boost::mpi::communicator parent;

  /**
   * Intercommunicators to children.
   */
  std::set<boost::mpi::communicator> children;
};
}

#endif
