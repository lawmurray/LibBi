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
  const MPI_Comm& getParent() const;

  /**
   * Set parent.
   */
  void setParent(const MPI_Comm& comm);

  /**
   * Get children.
   */
  std::set<MPI_Comm>& getChildren();

  /**
   * Add child.
   *
   * @param comm Communicator associated with the child.
   *
   * @return Number of children before adding this new child.
   *
   * Queue the child to be added on the next call to updateChildren().
   * Thread safe.
   */
  int addChild(const MPI_Comm& comm);

  /**
   * Remove child.
   *
   * @param comm Communicator associated with the child.
   *
   * Queue the child to be removed on the next call to updateChildren().
   * Thread safe.
   */
  void removeChild(const MPI_Comm& comm);

  /**
   * Update children list.
   *
   * @return Number of children after the update.
   *
   * Update the children list according to prior calls made to addChild() and
   * removeChild(). Thread safe.
   */
  int updateChildren();

private:
  /**
   * Communicator to parent.
   */
  MPI_Comm parent;

  /**
   * Intercommunicators to children.
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

inline const MPI_Comm& bi::TreeNetworkNode::getParent() const {
  return parent;
}

inline std::set<MPI_Comm>& bi::TreeNetworkNode::getChildren() {
  return comms;
}

#endif
