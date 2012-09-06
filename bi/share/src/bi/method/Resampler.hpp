/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_RESAMPLER_HPP
#define BI_METHOD_RESAMPLER_HPP

#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "../misc/location.hpp"

namespace bi {
/**
 * @internal
 *
 * @tparam V1 Vector type.
 *
 * Select ancestor for each particle given offspring.
 */
template<class V1>
struct resample_ancestors : public std::unary_function<
    thrust::tuple<int,int,int>,void> {
  /**
   * Ancestors vector for output.
   *
   * @note nvcc will not permit a device_vector here, as it considers
   * operator[] to be an external function call it seems.
   */
  int* as;

  /**
   * Number of particles.
   */
  const int P;

  /**
   * Constructor.
   */
  CUDA_FUNC_HOST resample_ancestors(V1 as) : as(thrust::raw_pointer_cast(&as[0])), P(as.size()) {
    //
  }

  /**
   * Apply functor.
   *
   * @param Particle index.
   */
  CUDA_FUNC_BOTH void operator()(thrust::tuple<int,int,int> x) {
    const int& p = thrust::get<0>(x);
    const int& o = thrust::get<1>(x);
    const int& O = thrust::get<2>(x);
    int i;

    for (i = 0; i < o && O + i < P; ++i) {
      as[O + i] = p;
    }
  }
};

/**
 * @internal
 *
 * Determine error in particular resampling.
 */
template<class T>
struct resample_check : public std::binary_function<T,int,T> {
  const T lW;
  const T P;
  // ^ oddly, casting o or P in operator()() causes a hang with CUDA 3.1 on
  //   Fermi, so we set the type of P to T instead of int

  /**
   * Constructor.
   */
  CUDA_FUNC_HOST resample_check(const T lW, const int P) : lW(lW),
      P(P) {
    //
  }

  /**
   * Apply functor.
   *
   * @param lw Log-weight for this index.
   * @param o Number of offspring for this index.
   *
   * @return Contribution to error for this index.
   */
  CUDA_FUNC_BOTH T operator()(const T& lw, const int& o) {
    T eps;

    if (BI_IS_FINITE(lw)) {
      eps = bi::exp(lw - lW) - o/P; // P of type T, not int, see note above
      eps *= eps;
    } else {
      eps = 0.0;
    }

    return eps;
  }
};

/**
 * @internal
 *
 * Resampler implementation on device.
 */
class ResamplerDeviceImpl {
public:
  /**
   * @copydoc Resampler::permute()
   */
  template<class V1>
  static void permute(V1 as);

  /**
   * @copydoc Resampler::copy()
   */
  template<class V1, class M1>
  static void copy(const V1 as, M1 s);
};

/**
 * @internal
 *
 * Resampler implementation on host.
 */
class ResamplerHostImpl {
public:
  /**
   * @copydoc Resampler::permute()
   */
  template<class V1>
  static void permute(V1 as) {
    /* pre-condition */
    BI_ASSERT(!V1::on_device);

    typename V1::size_type i;
    typename V1::value_type j;
    for (i = 0; i < as.size(); ++i) {
      if (as[i] != i && as[i] < as.size() && as[as[i]] != as[i]) {
        /* swap */
        j = as[as[i]];
        as[as[i]] = as[i];
        as[i] = j;
        --i; // repeat for new value
      }
    }
  }

  /**
   * @copydoc Resampler::copy()
   */
  template<class V1, class M1>
  static void copy(const V1 as, M1 X) {
    /* pre-condition */
    BI_ASSERT(!V1::on_device);
    BI_ASSERT(!M1::on_device);
    BI_ASSERT(as.size() <= X.size1());

    int p;
    for (p = 0; p < as.size(); ++p) {
      if (as[p] != p) {
        row(X, p) = row(X, as[p]);
      }
    }
  }
};

/**
 * %Resampler for particle filter.
 *
 * @ingroup method
 */
class Resampler {
public:
  /**
   * @name Low-level interface.
   */
  //@{
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
   * Permute ancestors to permit in-place copy.
   *
   * @tparam V1 Integral vector type.
   *
   * @param[in,out] as Ancestry.
   */
  template<class V1>
  static void permute(V1 as);

  /**
   * Correct weights after resampling with proposal.
   *
   * @tparam V1 Integral vector type.
   * @tparam V2 Vector type.
   * @tparam V2 Vector type.
   *
   * @param as Ancestry.
   * @param qlws Proposal log-weights.
   * @param[in,out] lws Log-weights.
   *
   * Assuming that a resample has been performed using the weights @p qlws,
   * The weights @p lws are set as importance weights, such that if
   * \f$a^i = p\f$, \f$w^i = w^p/w^{*p}\f$, where \f$w^{*p}\f$ are the
   * proposal weights (@p qlws) and \f$w^p\f$ the particle weights (@p lws).
   */
  template<class V1, class V2, class V3>
  static void correct(const V1 as, const V2 qlws, V3 lws);

  /**
   * In-place copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param as Ancestry.
   * @param[in,out] X Matrix. Rows of the matrix are copied.
   *
   * The copy is performed in-place. For each particle @c i that is to be
   * preserved (i.e. its offspring count is at least 1), @c a[i] should equal
   * @c i. This ensures that all particles are either read or (over)written,
   * but not both. Use permute() to ensure that an ancestry satisfies this
   * constraint.
   */
  template<class V1, class M1>
  static void copy(const V1 as, M1 X);

  /**
   * In-place copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param as Ancestry.
   * @param[in,out] s State.
   */
  template<class V1, class B, Location L>
  static void copy(const V1 as, State<B,L>& s);

  /**
   * In-place copy based on ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam T1 Assignable type.
   *
   * @param as Ancestry,
   * @oaram[in,out] v STL vector.
   */
  template<class V1, class T1>
  static void copy(const V1 as, std::vector<T1*>& v);

  /**
   * Compute effective sample size (ESS) of log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @tparam lws Log-weights.
   *
   * @return ESS.
   */
  template<class V1>
  static typename V1::value_type ess(const V1 lws);

  /**
   * Compute squared error of ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   *
   * @param lws Log-weights.
   * @param os Offspring.
   *
   * @return Squared error.
   *
   * This computes the squared error in the resampling, as in
   * @ref Kitagawa1996 "Kitagawa (1996)":
   *
   * \f[
   * \xi = \sum_{i=1}^P \left(\frac{o_i}{P} - \frac{w_i}{W}\right)^2\,,
   * \f]
   *
   * where \f$W\f$ is the sum of weights.
   */
  template<class V1, class V2>
  static typename V1::value_type error(const V1 lws, const V2 os);

  /**
   * Compute log-likelihood of ancestry.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   *
   * @param lws Log-weights.
   * @param os Offspring.
   *
   * @return Log-likelihood of the offspring under the multinomial
   * distribution defined by the weights.
   */
  template<class V1, class V2>
  static typename V1::value_type loglikelihood(const V1 lws, const V2 os);
  //@}
};

}

#include "../primitive/vector_primitive.hpp"
#include "../math/temp_vector.hpp"
#include "../math/sim_temp_vector.hpp"

#include "thrust/inner_product.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/scan.h"
#include "thrust/for_each.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/fill.h"
#include "thrust/gather.h"
#include "thrust/binary_search.h"
#include "thrust/extrema.h"

#include "boost/typeof/typeof.hpp"
#include "boost/mpl/if.hpp"

template<class V1, class V2>
void bi::Resampler::offspringToAncestors(const V1 os, V2 as) {
  /* pre-conditions */
  BI_ASSERT(sum_reduce(os) == as.size());

  typename sim_temp_vector<V1>::type Os(os.size());
  thrust::exclusive_scan(os.begin(), os.end(), Os.begin());
  BOOST_AUTO(first, thrust::make_zip_iterator(thrust::make_tuple(
      thrust::counting_iterator<int>(0),
      os.begin(),
      Os.begin())));
  BOOST_AUTO(last, thrust::make_zip_iterator(thrust::make_tuple(
      thrust::counting_iterator<int>(os.size()),
      os.end(),
      Os.end())));
  thrust::for_each(first, last, resample_ancestors<V2>(as));
}

template<class V1, class V2>
void bi::Resampler::ancestorsToOffspring(const V1 as, V2 os) {
  /* pre-conditions */
  BI_ASSERT(os.size() == as.size());

  const int P = os.size();
  typename sim_temp_vector<V1>::type keys(2*P), values(2*P);

  /* keys will consist of ancestry in [0..P-1], and 0..P-1 in [P..2P-1],
   * ensuring that all particle indices are represented */
  thrust::copy(as.begin(), as.end(), keys.begin());
  thrust::sequence(keys.begin() + P, keys.end());

  /* values are 1 for indices originally from the ancestry, 0 for others */
  thrust::fill(values.begin(), values.begin() + P, 1);
  thrust::fill(values.begin() + P, values.end(), 0);

  /* sort all that by key */
  bi::sort_by_key(keys, values);

  /* reduce by key to get final offspring counts */
  thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(),
      keys.begin(), os.begin());

  /* post-condition */
  BI_ASSERT(thrust::reduce(os.begin(), os.end()) == P);
}

template<class V1>
void bi::Resampler::permute(V1 as) {
  typedef typename boost::mpl::if_c<V1::on_device,
      ResamplerDeviceImpl,
      ResamplerHostImpl>::type impl;
  impl::permute(as);
}

template<class V1, class V2, class V3>
void bi::Resampler::correct(const V1 as, const V2 qlws, V3 lws) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  typedef typename V3::value_type T3;
  const int P = as.size();

  typename sim_temp_vector<V3>::type num(P), den(P);

  thrust::gather(as.begin(), as.end(), lws.begin(), num.begin());
  thrust::gather(as.begin(), as.end(), qlws.begin(), den.begin());
  lws.resize(P);
  thrust::transform(num.begin(), num.end(), den.begin(), lws.begin(),
      thrust::minus<T3>());
}

template<class V1, class M1>
void bi::Resampler::copy(const V1 as, M1 s) {
  /* pre-condition */
  BI_ASSERT(as.size() <= s.size1());

  typedef typename boost::mpl::if_c<M1::on_device,
      ResamplerDeviceImpl,
      ResamplerHostImpl>::type impl;
  impl::copy(as, s);
}

template<class V1, class B, bi::Location L>
void bi::Resampler::copy(const V1 as, State<B,L>& s) {
  s.setRange(s.start(), bi::max(s.size(), as.size()));
  copy(as, s.getDyn());
  s.setRange(s.start(), as.size());
}

template<class V1, class T1>
void bi::Resampler::copy(const V1 as, std::vector<T1*>& v) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);

  int i;
  for (i = 0; i < as.size(); ++i) {
    if (i != as(i)) {
      v[i]->resize(v[as(i)]->size());
      *v[i] = *v[as(i)];
    }
  }
}

template<class V1>
typename V1::value_type bi::Resampler::ess(const V1 lws) {
  return ess_reduce(lws);
}

template<class V1, class V2>
typename V1::value_type bi::Resampler::error(const V1 lws, const V2 os) {
  real lW = log_sum_exp(lws.begin(), lws.end(), BI_REAL(0.0));

  return thrust::inner_product(lws.begin(), lws.end(), os.begin(), BI_REAL(0.0),
      thrust::plus<real>(), resample_check<real>(lW, lws.size()));
}

template<class V1, class V2>
typename V1::value_type bi::Resampler::loglikelihood(const V1 lws,
    const V2 os) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());

  typedef typename V1::value_type T1;

  const int P = lws.size();

  /* normalising constant */
  BOOST_AUTO(iter1, thrust::make_transform_iterator(
      thrust::counting_iterator<T1>(1.0), log_functor<real>()));
  BOOST_AUTO(iter2, thrust::make_transform_iterator(os.begin(),
      log_factorial_functor<T1>()));
  T1 logNum = thrust::reduce(iter1, iter1 + P);
  T1 logDen = thrust::reduce(iter2, iter2 + P);

  /* exponent */
  T1 lW = log_sum_exp(lws.begin(), lws.end(), BI_REAL(0.0)); // log total weight
  BOOST_AUTO(nlws, thrust::make_transform_iterator(lws.begin(),
      add_constant_functor<T1>(-lW))); // normalised log weights
  T1 expon = thrust::inner_product(os.begin(), os.end(), nlws, BI_REAL(0.0));

  return logNum - logDen + expon;
}

#ifdef __CUDACC__
#include "../cuda/method/Resampler.cuh"
#endif

#endif
