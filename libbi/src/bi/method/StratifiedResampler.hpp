/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_STRATIFIEDRESAMPLER_HPP
#define BI_METHOD_STRATIFIEDRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../math/vector.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

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
    if (W > BI_REAL(0.0) && Ws > BI_REAL(0.0)) {
      return static_cast<int>(Ws/W*n - a + BI_REAL(1.0));
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
   * @param sort True to pre-sort weights, false otherwise.
   */
  StratifiedResampler(const bool sort = true);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, class B, Location L>
  void resample(Random&, V1 lws, V2 as, State<B,L>& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, class B, Location L>
  void resample(Random&, const V1 qlws, V2 lws, V3 as, State<B,L>& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, class B, Location L>
  void resample(Random& rng, const int a, V1 lws, V2 as, State<B,L>& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, class B, Location L>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as,
      State<B,L>& s) throw (ParticleFilterDegeneratedException);
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
   * @param rng Random number generator.
   * @param lws Log-weights.
   * @param[out] o Offspring.
   * @param n Number of samples to draw.
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 o, const int n)
      throw (ParticleFilterDegeneratedException);
  //@}

private:
  /**
   * Pre-sort weights?
   */
  bool sort;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"
#include "../misc/location.hpp"
#include "../math/temp_vector.hpp"
#include "../math/sim_temp_vector.hpp"

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/extrema.h"
#include "thrust/transform.h"
#include "thrust/reduce.h"
#include "thrust/scan.h"
#include "thrust/transform_scan.h"
#include "thrust/for_each.h"
#include "thrust/adjacent_difference.h"

template<class V1, class V2, class B, bi::Location L>
void bi::StratifiedResampler::resample(Random& rng, V1 lws, V2 as,
    State<B,L>& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == as.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V2>::type os(P);

  offspring(rng, lws, os, P);
  ancestors(os, as);
  permute(as);
  lws.clear();
  copy(as, s);
}

template<class V1, class V2, class B, bi::Location L>
void bi::StratifiedResampler::resample(Random& rng, const int a, V1 lws,
    V2 as, State<B,L>& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  const int P = lws.size();
  typename sim_temp_vector<V2>::type os(P);

  offspring(rng, lws, os, P - 1);
  ++os[a];
  ancestors(os, as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class B, bi::Location L>
void bi::StratifiedResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, State<B,L>& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type os(P);

  offspring(rng, qlws, os, P);
  ancestors(os, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2, class V3, class B, bi::Location L>
void bi::StratifiedResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, State<B,L>& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type os(P);

  offspring(rng, qlws, os, P - 1);
  ++os[a];
  ancestors(os, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2>
void bi::StratifiedResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == os.size());

  typedef typename V1::value_type T1;

  const int P = lws.size();
  typename sim_temp_vector<V1>::type lws1(P), Ws(P);
  typename sim_temp_vector<V2>::type Os(P), ps(P);
  T1 W, a;
  lws1 = lws;

  if (sort) {
    seq_elements(ps, 0);
    bi::sort_by_key(lws1, ps);
  }
  bi::inclusive_scan_sum_exp(lws1, Ws);

  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());
    if (sort) {
      bi::sort_by_key(ps, os);
    }

    #ifndef NDEBUG
    int m = thrust::reduce(os.begin(), os.end());
    BI_ASSERT(m == n, "Stratified resampler gives " << m << " offspring, should give " << n);
    #endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

#endif
