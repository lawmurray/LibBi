/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "TreeNetworkNode.hpp"

void bi::TreeNetworkNode::setParent(MPI_Comm& comm) {
  parent = comm;
}

int bi::TreeNetworkNode::addChild(MPI_Comm& comm) {
  int n = 0;
#pragma omp critical(TreeNetworkNode)
  {
    n = comms.size() + newcomms.size();
    newcomms.insert(comm);
  }
  return n;
}

void bi::TreeNetworkNode::removeChild(MPI_Comm& comm) {
#pragma omp critical(TreeNetworkNode)
  {
    oldcomms.insert(comm);
  }
}

int bi::TreeNetworkNode::updateChildren() {
  int n = 0;
#pragma omp critical(TreeNetworkNode)
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
