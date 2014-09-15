/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "TreeNetworkNode.hpp"

#include "../misc/assert.hpp"

bi::TreeNetworkNode::TreeNetworkNode() :
    parent(MPI_COMM_NULL, boost::mpi::comm_attach) {
  //
}

inline boost::mpi::communicator bi::TreeNetworkNode::getParent() const {
  boost::mpi::communicator tmp;

#pragma omp critical(TreeNetworkNode)
  tmp = parent;

  return tmp;
}

void bi::TreeNetworkNode::setParent(boost::mpi::communicator comm) {
#pragma omp critical(TreeNetworkNode)
  parent = comm;
}

inline std::set<boost::mpi::communicator> bi::TreeNetworkNode::getChildren() {
  std::set < boost::mpi::communicator > tmp;

#pragma omp critical(TreeNetworkNode)
  tmp = children;

  return tmp;
}

int bi::TreeNetworkNode::addChild(boost::mpi::communicator comm) {
  int n = 0;
#pragma omp critical(TreeNetworkNode)
  {
    n = children.size();
    children.insert(comm);
  }
  return n;
}

int bi::TreeNetworkNode::removeChild(boost::mpi::communicator comm) {
  int n = 0;
#pragma omp critical(TreeNetworkNode)
  {
    children.erase(comm);
    n = children.size();
  }
  return n;
}
