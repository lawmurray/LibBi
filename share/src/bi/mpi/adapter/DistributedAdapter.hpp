/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_ADAPTER_DISTRIBUTEDADAPTER_HPP
#define BI_MPI_ADAPTER_DISTRIBUTEDADAPTER_HPP

#include "../TreeNetworkNode.hpp"
#include "../mpi.hpp"

namespace bi {
/**
 * Distributed adapter.
 *
 * @ingroup adapter
 *
 * @tparam A Adapter type.
 *
 * DistributedAdapter is designed to work with a client-server architecture.
 * The generic implementation merely passes samples and weights added in a
 * client process onto the server process. This generic approach works in all
 * cases. There is scope to explicitly implement specialisations of the
 * class template for particular adapter types in order to perform some share
 * of aggregation on the client to reduce message sizes.
 */
template<class A>
class DistributedAdapter {
public:
  /**
   * Constructor.
   *
   * @param base Base adapter.
   * @param node Network node.
   */
  DistributedAdapter(A& base, TreeNetworkNode& node);

  /**
   * Destructor.
   */
  ~DistributedAdapter();

  /**
   * @copydoc Adapter::adapt()
   */
  template<class Q1>
  void adapt(const int k, Q1& q);

  /**
   * @copydoc Adapter::add()
   */
  template<class V1, class V2>
  void add(const V1 x, const V2 lws);

  /**
   * @copydoc Adapter::reset()
   */
  void reset();

private:
  /**
   * Send buffer up to parent.
   */
  void send();

  /**
   * Finish sends.
   */
  void finish();

  /**
   * Network node.
   */
  TreeNetworkNode& node;

  /**
   * Request handle for sending samples to parent.
   */
  boost::mpi::request requestSamples;

  /**
   * Request handle for sending log-weights to parent.
   */
  boost::mpi::request requestLogWeights;

  /**
   * Cache of samples currently being sent.
   */
  Cache2D<real> cacheSendSamples;

  /**
   * Cache of log-weights currently being sent.
   */
  Cache2D<real> cacheSendLogWeights;

  /**
   * Cache of samples currently being accumulated.
   */
  Cache2D<real> cacheAccumSamples;

  /**
   * Cache of log-weights currently being accumulated.
   */
  Cache2D<real> cacheAccumLogWeights;

  /**
   * Index into send cache for next write.
   */
  int pSend;

  /**
   * Index into accumulation cache for next write.
   */
  int pAccum;

  /**
   * Maximum number of accumulated weights before blocking.
   */
  static const int MAX_ACCUM = 1024;
};
}

template<class A>
bi::DistributedAdapter<A>::DistributedAdapter(A& base, TreeNetworkNode& node) :
    node(node), pSend(0), pAccum(0) {
  //
}

template<class A>
bi::DistributedAdapter<A>::~DistributedAdapter() {
  finish();
}

template<class A>
template<class V1, class V2>
void bi::DistributedAdapter<A>::add(const V1 x, const V2 lws) {
  cacheAccumSamples.set(pAccum, x);
  cacheAccumLogWeights.set(pAccum, lws);
  ++pAccum;
  send();
}

template<class A>
void bi::DistributedAdapter<A>::reset() {
  finish();
  cacheSendSamples.clear();
  cacheSendLogWeights.clear();
  cacheAccumSamples.clear();
  cacheAccumLogWeights.clear();
  pSend = 0;
  pAccum = 0;
}

template<class A>
void bi::DistributedAdapter<A>::send() {
  if (node.parent != MPI_COMM_NULL) {
    bool flag = true;
    if (pAccum == MAX_ACCUM) {
      requestSamples.wait();
      requestLogWeights.wait();
    } else {
      flag = requestSamples.test() && requestLogWeights.test();
    }
    if (flag) {
      cacheSendSamples.swap(cacheAccumSamples);
      cacheSendLogWeights.swap(cacheAccumLogWeights);
      std::swap(pSend, pAccum);

      cacheAccumSamples.clear();
      cacheAccumLogWeights.clear();

      BOOST_AUTO(X, cacheSendSamples.get(0, pSend));
      BOOST_AUTO(Lws, cacheSendLogWeights.get(0, pSend));
      BI_ASSERT(X.contiguous());
      BI_ASSERT(Lws.contiguous());

      requestSamples = node.parent.isend(0, MPI_TAG_ADAPTER_SAMPLES, X.buf(),
          X.size1() * X.size2());
      requestLogWeights = node.parent.isend(0, MPI_TAG_ADAPTER_LOGWEIGHTS,
          Lws.buf(), Lws.size1() * Lws.size2());
    }
  }
}

template<class A>
void bi::DistributedAdapter<A>::finish() {
  /* finish outstanding send */
  requestSamples.wait();
  requestLogWeights.wait();

  /* send any remaining */
  send();
  requestSamples.wait();
  requestLogWeights.wait();
}

#endif
