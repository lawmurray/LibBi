/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_MPI_ADAPTER_DISTRIBUTEDADAPTER_HPP
#define BI_MPI_ADAPTER_DISTRIBUTEDADAPTER_HPP

#include "../../adapter/Adapter.hpp"

namespace bi {
/**
 * Distributed adapter.
 *
 * @ingroup method_adapter
 */
template<class A>
class DistributedAdapter: public Adapter<A> {
public:
  /**
   * Constructor.
   */
  DistributedAdapter(const bool local = false, const double scale = 0.25,
      const double essRel = 0.25);

  template<class S1>
  bool adapt(const S1& s);
};
}

template<class A>
bi::DistributedAdapter<A>::DistributedAdapter(const bool local,
    const double scale, const double essRel) :
    Adapter<A>(local, scale, essRel) {
  //
}

template<class A>
template<class S1>
bool bi::DistributedAdapter<A>::adapt(const S1& s) {
  return A::distributedAdapt(s);
}

#endif
