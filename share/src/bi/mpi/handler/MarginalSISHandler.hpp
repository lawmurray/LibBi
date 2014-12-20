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
 * @tparam B Model type.
 * @tparam A Adapter type.
 * @tparam S Stopper type.
 */
template<class B, class A, class S>
class MarginalSISHandler {
public:
  /**
   * Constructor.
   *
   * @param B Model.
   * @param T Number of observations.
   * @param adapter Adapter.
   * @param stopper Stopper.
   * @param node Network node.
   */
  MarginalSISHandler(B& m, const int T, A& adapter, S& stopper,
      TreeNetworkNode& node);

  /**
   * Is all work complete?
   */
  bool done() const;

  /**
   * Send required initialisation data to child upon joining the computation.
   *
   * @param child Intercommunicator associated with the child.
   */
  void init(boost::mpi::communicator child);

  /**
   * Handle message from a child.
   *
   * @param child Intercommunicator associated with the child.
   * @param status Status of the probe that detected the message.
   */
  void handle(boost::mpi::communicator child, boost::mpi::status status);

private:
  /*
   * Handlers for specific events.
   */
  void handleStopperLogWeights(boost::mpi::communicator child,
      boost::mpi::status status);
  void handleAdapterSamples(boost::mpi::communicator child,
      boost::mpi::status status);

  /**
   * Model.
   */
  B& m;

  /**
   * Current observation.
   */
  int t;

  /**
   * Number of observations.
   */
  const int T;

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

template<class B, class A, class S>
bi::MarginalSISHandler<B,A,S>::MarginalSISHandler(B& m, const int T, A& adapter,
    S& stopper, TreeNetworkNode& node) :
    m(m), t(0), T(T), adapter(adapter), stopper(stopper), node(node) {
  //
}

template<class B, class A, class S>
bool bi::MarginalSISHandler<B,A,S>::done() const {
  /* stopping criterion reached, all children have return their outputs and
   * disconnected */
  return t == T && stopper.stop() && node.children.empty();
}

template<class B, class A, class S>
void bi::MarginalSISHandler<B,A,S>::init(boost::mpi::communicator child) {
  BOOST_AUTO(q, adapter.get(t));
  node.requests.push_front(child.isend(0, MPI_TAG_ADAPTER_PROPOSAL, q));
}

template<class B, class A, class S>
void bi::MarginalSISHandler<B,A,S>::handle(boost::mpi::communicator child,
    boost::mpi::status status) {
  switch (status.tag()) {
  case MPI_TAG_STOPPER_LOGWEIGHTS:
    handleStopperLogWeights(child, status);
    break;
  case MPI_TAG_ADAPTER_SAMPLES:
    handleAdapterSamples(child, status);
    break;
  default:
    BI_WARN_MSG(false,
        "Misbehaving child, out-of-sequence tag " << status.tag());
  }
}

template<class B, class A, class S>
void bi::MarginalSISHandler<B,A,S>::handleStopperLogWeights(
    boost::mpi::communicator child, boost::mpi::status status) {
  typedef typename temp_host_vector<real>::type vector_type;

  double maxlw = BI_INF;

  /* add weights */
  boost::optional<int> n = status.template count<real>();
  if (n) {
    vector_type lws(*n);
    child.recv(status.source(), status.tag(), lws.buf(), *n);
    stopper.add(lws, maxlw);
  }

  /* signal stop if necessary */
  if (stopper.stop()) {
    BOOST_AUTO(iter, node.children.begin());
    for (; iter != node.children.end(); ++iter) {
      node.requests.push_front(iter->isend(0, MPI_TAG_STOPPER_STOP));
    }
  }
}

template<class B, class A, class S>
void bi::MarginalSISHandler<B,A,S>::handleAdapterSamples(
    boost::mpi::communicator child, boost::mpi::status status) {
  typedef typename temp_host_matrix<real>::type matrix_type;

  static const int N = B::NP;

  /* add samples */
  boost::optional<int> n = status.template count<real>();
  if (n) {
    matrix_type Z(N + T, *n / (N + T));
    child.recv(status.source(), status.tag(), Z.buf(), *n);

    for (int j = 0; j < Z.size2(); ++j) {
      adapter.add(subrange(column(Z,j), 0, N), subrange(column(Z,j), N, T));
    }
  }

  /* send new proposal if necessary */
  if (adapter.stop(t)) {
    adapter.adapt(t);
    BOOST_AUTO(q, adapter.get(t));
    BOOST_AUTO(iter, node.children.begin());
    for (; iter != node.children.end(); ++iter) {
      node.requests.push_front(iter->isend(0, MPI_TAG_ADAPTER_PROPOSAL, q));
    }
    ///@todo Serialize q into archive just once, then send to all. This may
    ///be how broadcast is already implemented in Boost.MPI.
  }
}

#endif
