/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_MISC_HPP
#define BI_RESAMPLER_MISC_HPP

namespace bi {
/**
 * Compute offspring vector from ancestors vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param as Ancestors.
 * @param[out] os Offspring.
 */
template<class V1, class V2>
static void ancestorsToOffspring(const V1 as, V2 os);

/**
 * Compute ancestor vector from offspring vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param os Offspring.
 * @param[out] as Ancestors.
 */
template<class V1, class V2>
static void offspringToAncestors(const V1 os, V2 as);

/**
 * Compute already-permuted ancestor vector from offspring vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param os Offspring.
 * @param[out] as Ancestors.
 */
template<class V1, class V2>
static void offspringToAncestorsPermute(const V1 os, V2 as);

/**
 * Compute ancestor vector from cumulative offspring vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param Os Cumulative offspring.
 * @param[out] as Ancestors.
 */
template<class V1, class V2>
static void cumulativeOffspringToAncestors(const V1 Os, V2 as);

/**
 * Compute already-permuted ancestor vector from cumulative offspring
 * vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param Os Cumulative offspring.
 * @param[out] as Ancestors.
 */
template<class V1, class V2>
static void cumulativeOffspringToAncestorsPermute(const V1 Os, V2 as);

/**
 * Compute offspring vector from cumulative offspring vector.
 *
 * @tparam V1 Integral vector type.
 * @tparam V2 Integral vector type.
 *
 * @param Os Cumulative offspring.
 * @param[out] os Offspring.
 */
template<class V1, class V2>
static void cumulativeOffspringToOffspring(const V1 Os, V2 os);

/**
 * Permute ancestors to permit in-place copy.
 *
 * @tparam V1 Integral vector type.
 *
 * @param[in,out] as Ancestry.
 */
template<class V1>
static void permute(V1 as);
}

#include "../host/resampler/ResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/ResamplerGPU.cuh"
#endif
#include "../primitive/vector_primitive.hpp"

template<class V1, class V2>
void bi::ancestorsToOffspring(const V1 as, V2 os) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::ancestorsToOffspring(as, os);
}

template<class V1, class V2>
void bi::offspringToAncestors(const V1 os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::offspringToAncestors(os, as);
}

template<class V1, class V2>
void bi::offspringToAncestorsPermute(const V1 os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::offspringToAncestorsPermute(os, as);
}

template<class V1, class V2>
void bi::cumulativeOffspringToAncestors(const V1 Os, V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2>
void bi::cumulativeOffspringToAncestorsPermute(const V1 Os,
    V2 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::cumulativeOffspringToAncestorsPermute(Os, as);
}

template<class V1, class V2>
static void bi::cumulativeOffspringToOffspring(const V1 Os, V2 os) {
  adjacent_difference(Os, os);
}

template<class V1>
void bi::permute(const V1 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,ResamplerGPU,ResamplerHost>::type impl;
#else
  typedef ResamplerHost impl;
#endif
  impl::permute(as);
}

#endif
