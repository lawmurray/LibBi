/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "DistributedAdapterFactory.hpp"

boost::shared_ptr<bi::DistributedAdapter<bi::GaussianAdapter> > bi::DistributedAdapterFactory::createGaussianAdapter(
    const bool local, const double scale, const double essRel) {
  return boost::make_shared < DistributedAdapter<GaussianAdapter>
      > (local, scale, essRel);
}
