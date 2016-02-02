/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_ADAPTER_ADAPTERFACTORY_HPP
#define BI_ADAPTER_ADAPTERFACTORY_HPP

#include "Adapter.hpp"
#include "GaussianAdapter.hpp"

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

namespace bi {
/**
 * Adapter factory.
 *
 * @ingroup method_resampler
 */
class AdapterFactory {
public:
  /**
   * Create Gaussian adapter.
   */
  static boost::shared_ptr<Adapter<GaussianAdapter> > createGaussianAdapter(
      const bool local = false, const double scale = 0.25,
      const double essRel = 0.5);
};
}

#endif
