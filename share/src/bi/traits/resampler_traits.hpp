/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_RESAMPLER_TRAITS_HPP
#define BI_TRAITS_RESAMPLER_TRAITS_HPP

namespace bi {
/**
 * Precompute type for given resampler.
 *
 * @ingroup method_resampler
 *
 * @tparam R Resampler type.
 * @tparam L Location.
 */
template<class R, Location L>
struct precompute_type {
  typedef int type;
};

/**
 * Does resampler need a maximum log-weight?
 *
 * @ingroup method_resampler
 */
template<class R>
struct resampler_needs_max {
  static const bool value = false;
};
}

#endif
