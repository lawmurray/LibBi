/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_MPI_ADAPTER_DISTRIBUTEDADAPTERFACTORY_HPP
#define BI_MPI_ADAPTER_DISTRIBUTEDADAPTERFACTORY_HPP

#include "DistributedAdapter.hpp"
#include "../../adapter/GaussianAdapter.hpp"

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

namespace bi {
/**
 * Distributed adapter factory.
 *
 * @ingroup method_resampler
 */
class DistributedAdapterFactory {
public:
  /**
   * Create Gaussian adapter.
   */
  static boost::shared_ptr<DistributedAdapter<GaussianAdapter> > createGaussianAdapter(
      const bool local = false, const double scale = 0.25,
      const double essRel = 0.25);
};
}

#endif
