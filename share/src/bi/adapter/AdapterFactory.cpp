/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "AdapterFactory.hpp"

boost::shared_ptr<bi::Adapter<bi::GaussianAdapter> > bi::AdapterFactory::createGaussianAdapter(
    const bool local, const double scale, const double essRel) {
  return boost::make_shared < Adapter<GaussianAdapter>
      > (local, scale, essRel);
}
