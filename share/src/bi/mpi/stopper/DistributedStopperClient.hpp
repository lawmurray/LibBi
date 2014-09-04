/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPERCLIENT_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPERCLIENT_HPP

namespace bi {
/**
 * Distributed stopper client.
 *
 * @ingroup stopper
 *
 * @tparam S Stopper type.
 */
template<class S>
class DistributedStopperClient: public S {
public:
  BI_PASSTHROUGH_CONSTRUCTORS(DistributedStopperClient, S)

  /**
   * @copydoc Stopper::stop()
   */
  bool stop(const double maxlw) const;

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw);

  /**
   * @copydoc Stopper::add(const V1, const double)
   */
  template<class V1>
  void add(const V1 lws, const double maxlw);

  /**
   * @copydoc Stopper::reset()
   */
  void reset();
};
}

template<class S>
bool bi::DistributedStopperClient<S>::stop(const double maxlw) const {
  boost::optional<boost::mpi::status> status = comm.iprobe(0, MPI_TAG_STOPPER_STOP);
  if (status) {
    comm.recv(status->source(), status->tag());
    return true;
  }
  return false;
}

template<class S>
void bi::DistributedStopperClient<S>::add(const double lw, const double maxlw) {
  comm.isend(0, MPI_TAG_STOPPER_ADD_WEIGHT, lw);
  comm.isend(0, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
}

template<class S>
template<class V1>
void bi::DistributedStopperClient<S>::add(const V1 lws, const double maxlw) {
  comm.isend(0, MPI_TAG_STOPPER_ADD_WEIGHTS, lws);
  comm.isend(0, MPI_TAG_STOPPER_MAX_WEIGHT, maxlw);
}

template<class S>
void bi::DistributedStopperClient<S>::reset() {
  // should really be synchronous...
}

#endif
