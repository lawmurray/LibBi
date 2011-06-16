/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_EXACTExactStratifiedResampler_HPP
#define BI_METHOD_EXACTExactStratifiedResampler_HPP

#include "../cuda/cuda.hpp"
#include "../cuda/math/vector.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "Resampler.hpp"

namespace bi {
/**
 * @internal
 *
 * Determine number of offspring for each particle for ExactStratifiedResampler.
 */
template<class T>
struct resample_offspring_exact : public std::binary_function<T,T,int> {
  const T W;
  const int n, F;

  /**
   * Constructor.
   *
   * @param W Sum of weights.
   * @param n Number of samples to draw.
   * @param F Number of particle filters to maintain.
   */
  CUDA_FUNC_HOST resample_offspring_exact(const T W, const int n, const int F) :
      W(W), n(n), F(F) {
    //
  }

  /**
   * Apply functor.
   *
   * @param lw Log-weight for this index.
   * @param a Random number between 0 and 1, used to determine whether to
   * drop particle.
   *
   * @return Number of offspring for particle at this index.
   */
  CUDA_FUNC_BOTH int operator()(const T lw, const T a) {
    if (a < CUDA_POW(1.0 - CUDA_EXP(lw)/W, n*F)) {
      return 0;
    } else {
      return static_cast<int>(CUDA_CEIL(n*CUDA_EXP(lw)/W));
    }
  }
};

/**
 * Stratified resampler for particle filter.
 *
 * @ingroup method
 *
 * Determistic, stratified resampler based on the scheme of
 * @ref Kitagawa1996 "Kitagawa (1996)", without pre-sorting.
 */
class ExactStratifiedResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param rng Random number generator.
   */
  ExactStratifiedResampler(Random& rng);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(V1&, V2&)
   */
  template<class V1, class V2>
  void resample(V1& lws, V2& as);

  /**
   * @copydoc concept::Resampler::resample(const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3>
  void resample(const V1& qlws, V2& lws, V3& as);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, V1&, V2&)
   */
  template<class V1, class V2>
  void resample(const int a, V1& lws, V2& as);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3>
  void resample(const int a, const V1& qlws, V2& lws, V3& as);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select number of offspring for each particle.
   *
   * @tparam V1 Floating point vector type.
   * @tparam V2 Integral vector type.
   *
   * @param lws Log-weights.
   * @param[out] o Offspring.
   * @param n Number of samples to draw.
   */
  template<class V1, class V2>
  void offspring(const V1& lws, V2& o, const int n);
  //@}

private:
  /**
   * Random number generator.
   */
  Random& rng;
};
}

#include "../math/functor.hpp"
#include "../math/primitive.hpp"
#include "../math/locatable.hpp"
#include "../math/temp_vector.hpp"
#include "../cuda/math/temp_vector.hpp"

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/extrema.h"
#include "thrust/transform.h"
#include "thrust/reduce.h"
#include "thrust/scan.h"
#include "thrust/transform_scan.h"
#include "thrust/for_each.h"
#include "thrust/adjacent_difference.h"

template<class V1, class V2>
void bi::ExactStratifiedResampler::resample(V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  int P = lws.size();
  BOOST_AUTO(os, temp_vector<V1>(lws.size()));
  BOOST_AUTO(lws1, duplicate_vector<V1>(lws));

  offspring(*lws1, *os, P);
  P = bi::sum(os->begin(), os->end(), 0);
  lws.resize(P);
  as.resize(P);
  ancestors(*os, as);
  permute(as);

  BOOST_AUTO(lws1Iter, make_permutation_iterator(lws1->begin(), as.begin()));
  BOOST_AUTO(lws1End, lws1Iter + P);
  BOOST_AUTO(osIter, make_permutation_iterator(os->begin(), as.begin()));
  thrust::transform(lws1Iter, lws1End, osIter, lws.begin(), thrust::divides<real>());

  synchronize();
  delete os;
  delete lws1;
}

template<class V1, class V2>
void bi::ExactStratifiedResampler::resample(const int a, V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  BOOST_AUTO(os, temp_vector<V2>(as.size()));

  offspring(lws, *os, lws.size() - 1);
  ++(*os)[a];
  ancestors(*os, as);
  permute(as);
  lws.clear();

  synchronize();
  delete os;
}

template<class V1, class V2, class V3>
void bi::ExactStratifiedResampler::resample(const V1& qlws, V2& lws, V3& as) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  BOOST_AUTO(os, temp_vector<V2>(P));

  offspring(qlws, *os, P);
  ancestors(*os, as);
  permute(as);
  correct(as, qlws, lws);

  synchronize();
  delete os;
}

template<class V1, class V2, class V3>
void bi::ExactStratifiedResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  BOOST_AUTO(os, temp_vector<V2>(P));

  offspring(qlws, *os, P - 1);
  ++(*os)[a];
  ancestors(*os, as);
  permute(as);
  correct(as, qlws, lws);

  synchronize();
  delete os;
}

#include "StratifiedResampler.hpp"

template<class V1, class V2>
void bi::ExactStratifiedResampler::offspring(const V1& lws, V2& os,
    const int n) {
  /* pre-condition */
  assert (lws.size() == os.size());

//  BOOST_AUTO(as, host_temp_vector<real>(lws.size()));
//  rng.uniforms(*as);
//
//  real W = bi::sum_exp(lws.begin(), lws.end(), REAL(0.0));
//  if (W > 0) {
//    thrust::transform(lws.begin(), lws.end(), as->begin(), os.begin(),
//        resample_offspring_exact<real>(W, n, 2));
//  } else {
//    BI_WARN(W > 0, "Particle filter has degenerated");
//    bi::fill(os.begin(), os.end(), 0);
//  }
//
//  synchronize();
//  delete as;

  static BI_UNUSED const Location L2 = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename locatable_temp_vector<L2,T1>::type V3;
  typedef typename locatable_temp_vector<L2,T2>::type V4;

  const int P = lws.size();
  V3 lws1(P), Ws(P);
  V4 Os(P);
  T1 W, a;
  lws1 = lws;

  BOOST_AUTO(as, host_temp_vector<real>(lws.size()));
  BOOST_AUTO(os2, host_temp_vector<real>(os.size()));
  rng.uniforms(*as);
  bi::inclusive_scan_sum_exp(lws1.begin(), lws1.end(), Ws.begin());

  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(),
        resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());

    thrust::transform(lws.begin(), lws.end(), as->begin(), os2->begin(),
        resample_offspring_exact<real>(W, n, 10));
    for (int i = 0; i < os2->size(); ++i) {
      if ((*os2)(i) == 0) {
        os(i) = 0;
      } else if (os(i) == 0) {
        os(i) = 1;
      }
    }
  } else {
    BI_WARN(W > 0, "Particle filter has degenerated");
    bi::fill(os.begin(), os.end(), 0);
  }
  synchronize();

  delete as;
  delete os2;
}

#endif
