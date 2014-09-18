/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_HANDLER_MARGINALSISHANDLER_HPP
#define BI_MPI_HANDLER_MARGINALSISHANDLER_HPP

#include "../TreeNetworkNode.hpp"
#include "../mpi.hpp"

namespace bi {
/**
 * Marginal
 *
 * @ingroup server
 *
 * @tparam A Adapter type.
 * @tparam S Stopper type.
 */
template<class A, class S>
class MarginalSISHandler {
public:
  /**
   * Constructor.
   *
   * @param adapter Adapter.
   * @param stopper Stopper.
   * @param node Network node.
   */
  MarginalSISHandler(A& adapter, S& stopper, TreeNetworkNode& node);

  /**
   * Is all work complete?
   */
  bool done() const;

  /**
   * Send required initialisation data to child upon joining the computation.
   *
   * @param child Intercommunicator associated with the child.
   */
  void init(boost::mpi::communicator child)

  /**
   * Handle message from a child.
   *
   * @param child Intercommunicator associated with the child.
   * @param status Status of the probe that detected the message.
   */
  void handle(boost::mpi::communicator child, boost::mpi::status status);

private:
  /**
   * Adapter.
   */
  A& adapter;

  /**
   * Stopper.
   */
  S& stopper;

  /**
   * Network node.
   */
  TreeNetworkNode& node;
};
}

template<class A, class S>
bi::MarginalSISHandler<A,S>::MarginalSISHandler(A& adapter, S& stopper, TreeNetworkNode& node) :
    adapter(adapter), stopper(stopper), node(node) {
  //
}

template<class A, class S>
bool bi::MarginalSISHandler<A,S>::done() const {

}

template<class A, class S>
void bi::MarginalSISHandler<A,S>::init(boost::mpi::communicator child) {
  //
}

template<class A, class S>
void bi::MarginalSISHandler<A,S>::handle(boost::mpi::communicator child,
    boost::mpi::status status) {
  typedef typename host_temp_vector<real>::type vector_type;

  double maxlw = std::numeric_limits < real > ::infinity();

  switch (status.tag()) {
  case MPI_TAG_STOPPER_ADD_WEIGHTS:
    boost::mpi::optional<int> n = status.template count<real>();
    if (n) {
      vector_type lws(*n);
      child.recv(status.source(), status.tag(), lws.buf(), lws.size());
      #pragma omp task
      stopper.add(lws, maxlw);
    }
    break;
  case MPI_TAG_STOPPER_ADD_WEIGHT:
    real lw;
    child.recv(status.source(), status.tag(), lw);
    #pragma omp task
    stopper.add(lw, maxlw);
    break;
  default:
    BI_WARN_MSG(false,
        "Misbehaving child, out-of-sequence tag " << status.tag());
  }

  if (stopper.stop()) {
    BOOST_AUTO(iter, node.children.begin());
    for (; iter != node.children.end(); ++iter) {
      iter->isend(0, MPI_TAG_STOPPER_STOP);
    }
  }
}

#endif
