/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_ADAPTER_CLIENTSERVERADAPTER_HPP
#define BI_MPI_ADAPTER_CLIENTSERVERADAPTER_HPP

#include "../mpi.hpp"
#include "../TreeNetworkNode.hpp"
#include "../../cache/Cache2D.hpp"

namespace bi {
/**
 * Distributed adapter.
 *
 * @ingroup adapter
 *
 * @tparam A Adapter type.
 *
 * ClientServerAdapter is designed to work with a client-server architecture.
 * The generic implementation merely passes samples and weights added in a
 * client process onto the server process. This generic approach works in all
 * cases. There is scope to explicitly implement specialisations of the
 * class template for particular adapter types in order to perform some share
 * of aggregation on the client to reduce message sizes.
 */
template<class A>
class ClientServerAdapter {
public:
  typedef typename A::proposal_type proposal_type;

  /**
   * Constructor.
   *
   * @param base Base adapter.
   * @param node Network node.
   */
  ClientServerAdapter(A& base, TreeNetworkNode& node);

  /**
   * Destructor.
   */
  ~ClientServerAdapter();

  /**
   * @copydoc Adapter::add()
   */
  template<class V1, class V2>
  void add(const V1 x, const V2 lws);

  /**
   * @copydoc Adapter::stop()
   */
  bool stop(const int k);

  /**
   * @copydoc Adapter::adapt()
   */
  void adapt(const int k);

  /**
   * @copydoc Adapter::get()
   */
  proposal_type& get(const int k);

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
   * Base adapter.
   */
  A& base;

  /**
   * Network node.
   */
  TreeNetworkNode& node;

  /**
   * Request handle for sends to parent.
   */
  boost::mpi::request request;

  /**
   * Cache of log-weights currently being sent.
   */
  Cache2D<real> cacheSend;

  /**
   * Cache of log-weights currently being accumulated.
   */
  Cache2D<real> cacheAccum;

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
bi::ClientServerAdapter<A>::ClientServerAdapter(A& base, TreeNetworkNode& node) :
    base(base), node(node), pSend(0), pAccum(0) {
  //
}

template<class A>
bi::ClientServerAdapter<A>::~ClientServerAdapter() {
  finish();
}

template<class A>
template<class V1, class V2>
void bi::ClientServerAdapter<A>::add(const V1 x, const V2 lws) {
  /* combine into one vector */
  typename temp_host_vector<real>::type z(x.size() + lws.size());
  subrange(z, 0, x.size()) = x;
  subrange(z, x.size(), lws.size()) = lws;
  cacheAccum.set(pAccum, z);
  ++pAccum;
  send();
}

template<class A>
bool bi::ClientServerAdapter<A>::stop(const int k) {

}

template<class A>
void bi::ClientServerAdapter<A>::adapt(const int k) {
  node.parent.recv(0, MPI_TAG_ADAPTER_PROPOSAL, base.q);
}

template<class A>
typename bi::ClientServerAdapter<A>::proposal_type& bi::ClientServerAdapter<A>::get(
    const int k) {
  return base.get(k);
}

template<class A>
void bi::ClientServerAdapter<A>::reset() {
  finish();
  cacheSend.clear();
  cacheAccum.clear();
  pSend = 0;
  pAccum = 0;
}

template<class A>
void bi::ClientServerAdapter<A>::send() {
  if (node.parent != MPI_COMM_NULL) {
    bool flag = true;
    if (pAccum == MAX_ACCUM) {
      request.wait();
    } else {
      flag = request.test();
    }
    if (flag) {
      cacheSend.swap(cacheAccum);
      cacheAccum.clear();
      pSend = pAccum;
      pAccum = 0;

      BOOST_AUTO(Z, cacheSend.get(0, pSend));
      BI_ASSERT(Z.contiguous());

      request = node.parent.isend(0, MPI_TAG_ADAPTER_SAMPLES, Z.buf(),
          Z.size1() * Z.size2());
    }
  }
}

template<class A>
void bi::ClientServerAdapter<A>::finish() {
  /* finish outstanding send */
  request.wait();

  /* send any remaining */
  send();
  request.wait();
}

#endif
