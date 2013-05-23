/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_MISC_HPP
#define BI_CUDA_RESAMPLER_MISC_HPP

namespace bi {
/**
 * Placeholder for passing as argument to kernels to enable pre-permute.
 */
enum EnablePrePermute {
  ENABLE_PRE_PERMUTE = 1
};

/**
 * Placeholder for passing as argument to kernels to disable pre-permute.
 */
enum DisablePrePermute {
  DISABLE_PRE_PERMUTE = 0
};

}

#endif
