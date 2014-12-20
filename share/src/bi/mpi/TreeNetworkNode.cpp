/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "TreeNetworkNode.hpp"

bi::TreeNetworkNode::TreeNetworkNode() :
    parent(MPI_COMM_NULL, boost::mpi::comm_attach) {
  //
}
