/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_STRATIFIEDRESAMPLER_HPP
#define BI_METHOD_STRATIFIEDRESAMPLER_HPP

#include "../cuda/cuda.hpp"
#include "../cuda/math/vector.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "Resampler.hpp"

namespace bi {
/**
 * @internal
 *
 * Determine number of offspring for each particle for StratifiedResampler.
 */
template<class T>
struct resample_offspring : public std::unary_function<T,int> {
  const T a, W;
  const int n;

  /**
   * Constructor.
   *
   * @param a Offset into strata.
   * @param W Sum of weights.
   * @param n Number of samples to draw.
   */
  CUDA_FUNC_HOST resample_offspring(const T a, const T W, const int n) :
      a(a), W(W), n(n) {
    //
  }

  /**
   * Apply functor.
   *
   * @param Ws Inclusive prefix sum of weights for this index.
   *
   * @return Number of offspring for particle at this index.
   */
  CUDA_FUNC_BOTH int operator()(const T Ws) {
    if (W > REAL(0.0) && Ws > REAL(0.0)) {
      return static_cast<int>(Ws/W*n - a + REAL(1.0));
    } else {
      return 0;
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
class StratifiedResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param rng Random number generator.
   * @param sort True to pre-sort weights, false otherwise.
   */
  StratifiedResampler(Random& rng, const bool sort = true);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const int a, const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);
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

  /**
   * Pre-sort weights?
   */
  bool sort;
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

template<class V1, class V2, bi::Location L>
void bi::StratifiedResampler::resample(V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());

  /* typically faster on host, so copy to there */
  BOOST_AUTO(lws1, host_map_vector(lws));
  BOOST_AUTO(as1, host_temp_vector<int>(lws.size()));
  BOOST_AUTO(os1, host_temp_vector<int>(lws.size()));

  if (V1::on_device) {
    synchronize();
  }
  offspring(*lws1, *os1, lws1->size());
  ancestors(*os1, *as1);
  permute(*as1);
  as = *as1;
  lws.clear();
  copy(as, theta, s);
  if (as.on_device) {
    synchronize();
  }
  delete lws1;
  delete as1;
  delete os1;
}

template<class V1, class V2, bi::Location L>
void bi::StratifiedResampler::resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  BOOST_AUTO(lws1, host_map_vector(lws));
  BOOST_AUTO(as1, host_temp_vector<int>(lws.size()));
  BOOST_AUTO(os1, host_temp_vector<int>(lws.size()));

  if (V1::on_device) {
    synchronize();
  }
  offspring(*lws1, *os1, lws1->size() - 1);
  ++(*os1)[a];
  ancestors(*os1, *as1);
  permute(*as1);
  as = *as1;
  copy(as, theta, s);
  lws.clear();

  if (V2::on_device) {
    synchronize();
  }
  delete lws1;
  delete as1;
  delete os1;
}

template<class V1, class V2, class V3, bi::Location L>
void bi::StratifiedResampler::resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  ///@todo Do on host, typically faster.

  /* pre-condition */
  const int P = as.size();
  assert (qlws.size() == lws.size());

  BOOST_AUTO(os, temp_vector<V2>(lws.size()));

  offspring(qlws, *os, P);
  ancestors(*os, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, theta, s);

  synchronize();
  delete os;
}

template<class V1, class V2, class V3, bi::Location L>
void bi::StratifiedResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  ///@todo Do on host, typically faster.

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
  copy(as, theta, s);

  synchronize();
  delete os;
}

template<class V1, class V2>
void bi::StratifiedResampler::offspring(const V1& lws, V2& os,
    const int n) {
  /* pre-condition */
  assert (lws.size() == os.size());

  static BI_UNUSED const Location L2 = V2::on_device ? ON_DEVICE : ON_HOST;
  typedef typename V1::value_type T1;
  typedef typename V2::value_type T2;
  typedef typename locatable_temp_vector<L2,T1>::type V3;
  typedef typename locatable_temp_vector<L2,T2>::type V4;

  const int P = lws.size();
  V3 lws1(P), Ws(P);
  V4 Os(P), ps(P);
  T1 W, a;
  lws1 = lws;

  if (sort) {
    bi::sequence(ps.begin(), ps.end(), 0);
    bi::sort_by_key(lws1.begin(), lws1.end(), ps.begin());
  }
  bi::inclusive_scan_sum_exp(lws1.begin(), lws1.end(), Ws.begin());

  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());
    if (sort) {
      bi::sort_by_key(ps.begin(), ps.end(), os.begin());
    }

    #ifndef NDEBUG
    int m = thrust::reduce(os.begin(), os.end());
    BI_WARN(m == n, "Stratified resampler gives " << m << " offspring, should give " << n);
    #endif
  } else {
    BI_WARN(W > 0, "Particle filter has degenerated");
    bi::fill(os.begin(), os.end(), 0);
  }
  synchronize();
}

#endif
