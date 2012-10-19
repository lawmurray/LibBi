/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_STRATIFIEDRESAMPLER_HPP
#define BI_RESAMPLER_STRATIFIEDRESAMPLER_HPP

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
 * @ingroup method_resampler
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
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, V1 lws, V2 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const V1 qlws, V2 lws, V3 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as,
      O1& s) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void cond_resample(Random& rng, const int ka, const int k, V1 lws, V2 as,
      O1& s) throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::offspring
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 o, const int P)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void offspring(Random& rng, const V1 lws, V2 o, const int n, int ka,
      bool sorted, V3 lws1, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void offspring(Random& rng, const V1 lws, V2 os,
      const int n, bool sorted, V3 lws1, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void ancestors(Random& rng, const V1 lws, V2 as, int P,
      bool sorted, V3 lws1, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void ancestors(Random& rng, const V1 lws, V2 as, int P, int ka, int k,
      bool sorted, V3 lws1, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);
  //@}

protected:
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

template<class V1, class V2, class O1>
void bi::StratifiedResampler::resample(Random& rng, V1 lws, V2 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  const int P = lws.size();
  typename sim_temp_vector<V2>::type os(P);

  offspring(rng, lws, os, P);
  offspringToAncestors(os, as);
  permute(as);
  lws.clear();
  copy(as, s);
}

template<class V1, class V2, class O1>
void bi::StratifiedResampler::cond_resample(Random& rng, const int ka,
    const int k, V1 lws, V2 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(k >= 0 && k < as.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());
  BI_ASSERT(k == 0 && ka == 0);

  const int P = lws.size();
  typename sim_temp_vector<V2>::type os(P);

  int P2;
  if (!sort) {
    // change this?
    P2 = 0;
  } else {
    P2 = s.size();
  }
  typename sim_temp_vector<V1>::type lws1(P2), Ws(P2);
  typename sim_temp_vector<V2>::type ps(P2);

  offspring(rng, lws, os, P, ka, false, lws1, ps, Ws);
  offspringToAncestors(os, as);
  permute(as);

  BI_ASSERT(*(as.begin() + k) == ka);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class O1>
void bi::StratifiedResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type os(P);

  offspring(rng, qlws, os, P);
  offspringToAncestors(os, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2, class V3, class O1>
void bi::StratifiedResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type os(P);

  offspring(rng, qlws, os, P - 1);
  ++os[a];
  offspringToAncestors(os, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2>
void bi::StratifiedResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());

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
  bi::inclusive_scan_sum_expu(lws1, Ws);

  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());
    if (sort) {
      typename sim_temp_vector<V2>::type temp(P);
      temp = os;
      bi::scatter(temp,ps,os);
    }

    #ifndef NDEBUG
    int m = thrust::reduce(os.begin(), os.end());
    BI_ASSERT_MSG(m == n, "Stratified resampler gives " << m << " offspring, should give " << n);
    #endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

template<class V1, class V2, class V3, class V4>
void bi::StratifiedResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());

  typedef typename V1::value_type T1;

  const int P = lws.size();
  typename sim_temp_vector<V2>::type Os(P);
  T1 W, a;

  if (sort) {
    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      bi::inclusive_scan_sum_expu(lws1, Ws);
    }
  } else {
    bi::inclusive_scan_sum_expu(lws, Ws);
  }

  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());
    if (sort) {
      typename sim_temp_vector<V2>::type temp(P);
      temp = os;
      bi::scatter(temp,ps,os);
    }

    #ifndef NDEBUG
    int m = thrust::reduce(os.begin(), os.end());
    BI_ASSERT_MSG(m == n, "Stratified resampler gives " << m << " offspring, should give " << n);
    #endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

// offspring for conditional SMC. May only work if ka = 0.
template<class V1, class V2, class V3, class V4>
void bi::StratifiedResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n, int ka, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());

  typedef typename V1::value_type T1;

  const int P = lws.size();
  typename sim_temp_vector<V2>::type Os(P);
  T1 W, a;

  if (sort) {
    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      bi::inclusive_scan_sum_expu(lws1, Ws);
    }
  } else {
    bi::inclusive_scan_sum_expu(lws, Ws);
  }

  W = *(Ws.end() - 1); // sum of weights

  int k = bi::find(ps,ka);
  real left = k > 0 ? *(Ws.begin() + k-1) : 0.0;
  real right = *(Ws.begin() + k);
  real c = rng.uniform(left,right);
  int strata = std::floor(n*c/W);
  a = n*c/W - strata;

  if (W > 0) {
//    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<T1>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());
    if (sort) {
      typename sim_temp_vector<V2>::type temp(P);
      temp = os;
      bi::scatter(temp,ps,os);
    }

    #ifndef NDEBUG
    int m = thrust::reduce(os.begin(), os.end());
    BI_ASSERT_MSG(m == n, "Stratified resampler gives " << m << " offspring, should give " << n);
    #endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

template<class V1, class V2>
void bi::StratifiedResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == lws.size());
  const int P = as.size();

  typename sim_temp_vector<V2>::type os(P), ps(P);
  typename sim_temp_vector<V1>::type lws1(P),Ws(P);

  offspring(rng, lws, os, P, false, lws1, ps, Ws);
  offspringToAncestors(os, as);
  permute(as);
}

template<class V1, class V2, class V3, class V4>
void bi::StratifiedResampler::ancestors(Random& rng, const V1 lws, V2 as, int P,
    bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typename sim_temp_vector<V2>::type os(lws.size());

  offspring(rng, lws, os, P, sorted, lws1, ps, Ws);
  offspringToAncestors(os, as);
  permute(as);
}

template<class V1, class V2, class V3, class V4>
void bi::StratifiedResampler::ancestors(Random& rng, const V1 lws, V2 as, int P, int ka,
    int k, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typename sim_temp_vector<V2>::type os(lws.size());

  offspring(rng, lws, os, P, ka, sorted, lws1, ps, Ws);
  offspringToAncestors(os, as);
  permute(as);

  BI_ASSERT(*(as.begin() + k)==ka);
}

#endif
