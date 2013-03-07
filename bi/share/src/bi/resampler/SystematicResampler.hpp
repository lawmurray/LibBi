/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_SYSTEMATICRESAMPLER_HPP
#define BI_RESAMPLER_SYSTEMATICRESAMPLER_HPP

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
 * Determine number of offspring for each particle for SystematicResampler.
 */
template<class T>
struct resample_cumulative_offspring: public std::unary_function<T,int> {
  const T a, W, n;

  /**
   * Constructor.
   *
   * @param a Relative offset into strata (between 0 and 1).
   * @param W Sum of weights.
   * @param n Number of samples to draw.
   */
  CUDA_FUNC_HOST
  resample_cumulative_offspring(const T a, const T W, const int n) :
      a(a), W(W), n(n) {
    /* pre-condition */
    BI_ASSERT(a >= 0.0 && a <= 1.0);
    BI_ASSERT(W > 0.0);
    BI_ASSERT(n > 0);
  }

  /**
   * Apply functor.
   *
   * @param Ws Inclusive prefix sum of weights for this index.
   *
   * @return Cumulative offspring for particle this index.
   */
  CUDA_FUNC_BOTH
  int operator()(const T Ws) {
    return bi::min(n, static_cast<int>(Ws / W * n + a));
  }
};

/**
 * Systematic resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Systematic (determistic stratified) resampler based on the scheme of
 * @ref Kitagawa1996 "Kitagawa (1996)", with optional pre-sorting.
 */
class SystematicResampler: public Resampler {
public:
  /**
   * Constructor.
   *
   * @param sort True to pre-sort weights, false otherwise.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   */
  SystematicResampler(const bool sort = true, const double essRel = 0.5);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  bool resample(Random& rng, V1 lws, V2 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  bool resample(Random& rng, const V1 qlws, V2 lws, V3 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  bool resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  bool cond_resample(Random& rng, const int ka, const int k, V1 lws, V2 as,
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
  void offspring(Random& rng, const V1 lws, V2 os, const int n, bool sorted,
      V3 lws1, V4 ps, V3 Ws) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::cumulativeoOffspring
   */
  template<class V1, class V2>
  void cumulativeOffspring(Random& rng, const V1 lws, V2 Os, const int P)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void cumulativeOffspring(Random& rng, const V1 lws, V2 Os, const int n,
      int ka, bool sorted, V3 lws1, V4 ps, V3 Ws)
          throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void cumulativeOffspring(Random& rng, const V1 lws, V2 Os, const int n,
      bool sorted, V3 lws1, V4 ps, V3 Ws)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void ancestors(Random& rng, const V1 lws, V2 as, int P, bool sorted,
      V3 lws1, V4 ps, V3 Ws) throw (ParticleFilterDegeneratedException);

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

template<class V1, class V2, class O1>
bool bi::SystematicResampler::resample(Random& rng, V1 lws, V2 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  const int P = lws.size();
  typename sim_temp_vector<V2>::type Os(P);

  bool r = isTriggered(lws);
  if (r) {
    cumulativeOffspring(rng, lws, Os, P);
    cumulativeOffspringToAncestorsPermute(Os, as);
    lws.clear();
    copy(as, s);
  } else {
    normalise(lws);
    seq_elements(as, 0);
  }
  return r;
}

template<class V1, class V2, class V3, class O1>
bool bi::SystematicResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type Os(P);

  bool r = isTriggered(lws);
  if (r) {
    cumulativeOffspring(rng, qlws, Os, P);
    cumulativeOffspringToAncestorsPermute(Os, as);
    correct(as, qlws, lws);
    normalise(lws);
    copy(as, s);
  } else {
    normalise(lws);
    seq_elements(as, 0);
  }
  return r;
}

template<class V1, class V2, class V3, class O1>
bool bi::SystematicResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(qlws.size() == lws.size());

  /* typically faster on host, so copy to there */
  const int P = lws.size();
  typename sim_temp_vector<V3>::type Os(P);

  bool r = isTriggered(lws);
  if (r) {
    cumulativeOffspring(rng, qlws, Os, P - 1);
    BOOST_AUTO(tail, subrange(Os, a, Os.size() - a));
    addscal_elements(tail, 1, tail);
    cumulativeOffspringToAncestorsPermute(Os, as);
    correct(as, qlws, lws);
    normalise(lws);
    copy(as, s);
  } else {
    normalise(lws);
    seq_elements(as, 0);
  }
  return r;
}

template<class V1, class V2, class O1>
bool bi::SystematicResampler::cond_resample(Random& rng, const int ka,
    const int k, V1 lws, V2 as, O1& s)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(k >= 0 && k < as.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());
  BI_ASSERT(k == 0 && ka == 0);

  bool r = isTriggered(lws);
  if (r) {
    const int P = lws.size();
    typename sim_temp_vector<V2>::type Os(P);

    int P2;
    if (!sort) {
      // change this?
      P2 = 0;
    } else {
      P2 = s.size();
    }
    typename sim_temp_vector<V1>::type lws1(P2), Ws(P2);
    typename sim_temp_vector<V2>::type ps(P2);

    cumulativeOffspring(rng, lws, Os, P, ka, false, lws1, ps, Ws);
    cumulativeOffspringToAncestorsPermute(Os, as);
    BI_ASSERT(*(as.begin() + k) == ka);
    copy(as, s);
    lws.clear();
  } else {
    normalise(lws);
    seq_elements(as, 0);
  }
  return r;
}

template<class V1, class V2>
void bi::SystematicResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();

  if (sort) {
    vector_type lws1(P), Ws(P);
    int_vector_type ps(P), Os(P), temp(P);

    lws1 = lws;
    seq_elements(ps, 0);
    bi::sort_by_key(lws1, ps);
    sumexpu_inclusive_scan(lws1, Ws);

    T1 W = *(Ws.end() - 1);  // sum of weights
    if (W > 0) {
      T1 a = rng.uniform((T1)0.0, (T1)1.0);  // offset into strata
      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));
      bi::adjacent_difference(Os, os);
      temp = os;
      bi::scatter(temp, ps, os);

#ifndef NDEBUG
      int m = sum_reduce(os);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  } else {
    int_vector_type Os(P);
    cumulativeOffspring(rng, lws, Os, n);
    bi::adjacent_difference(Os, os);
  }
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();

  if (sort) {
    int_vector_type Os(P), temp(P);

    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      sumexpu_inclusive_scan(lws1, Ws);
    }

    T1 W = *(Ws.end() - 1);  // sum of weights
    if (W > 0) {
      T1 a = rng.uniform((T1)0.0, (T1)1.0);  // offset into strata
      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));
      bi::adjacent_difference(Os, os);
      temp = os;
      bi::scatter(temp, ps, os);

#ifndef NDEBUG
      int m = sum_reduce(os);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  } else {
    int_vector_type Os(P);
    cumulativeOffspring(rng, lws, Os, n, sorted, lws1, ps, Ws);
    bi::adjacent_difference(Os, os);
  }
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int n, int ka, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /// @todo May only work if ka == 0

  /* pre-condition */
  BI_ASSERT(lws.size() == os.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();

  if (sort) {
    int_vector_type Os(P), temp(P);

    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      sumexpu_inclusive_scan(lws1, Ws);
    }

    T1 W = *(Ws.end() - 1);  // sum of weights

    if (W > 0) {
      int k = bi::find(ps, ka);
      real left = k > 0 ? *(Ws.begin() + k - 1) : 0.0;
      real right = *(Ws.begin() + k);
      real c = rng.uniform(left, right);
      int strata = std::floor(n * c / W);
      T1 a = n * c / W - strata;

      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));
      bi::adjacent_difference(Os, os);
      temp = os;
      bi::scatter(temp, ps, os);

#ifndef NDEBUG
      int m = sum_reduce(os);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  } else {
    int_vector_type Os(P);
    cumulativeOffspring(rng, lws, Os, n, ka, sorted, lws1, ps, Ws);
    bi::adjacent_difference(Os, os);
  }
}

template<class V1, class V2>
void bi::SystematicResampler::cumulativeOffspring(Random& rng, const V1 lws,
    V2 Os, const int n) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == Os.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();
  T1 W, a;

  if (sort) {
    int_vector_type os(P);
    offspring(rng, lws, os, n);
    sum_inclusive_scan(os, Os);
  } else {
    vector_type Ws(P);
    sumexpu_inclusive_scan(lws, Ws);

    W = *(Ws.end() - 1);  // sum of weights
    if (W > 0) {
      a = rng.uniform((T1)0.0, (T1)1.0);  // offset into strata

      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));

#ifndef NDEBUG
      int m = *(Os.end() - 1);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  }
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::cumulativeOffspring(Random& rng, const V1 lws,
    V2 Os, const int n, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == Os.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();

  if (sort) {
    int_vector_type os(P);
    offspring(rng, lws, os, n, sorted, lws1, ps, Ws);
    sum_inclusive_scan(os, Os);
  } else {
    sumexpu_inclusive_scan(lws, Ws);
    T1 W = *(Ws.end() - 1);  // sum of weights
    if (W > 0) {
      T1 a = rng.uniform((T1)0.0, (T1)1.0);  // offset into strata
      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));

#ifndef NDEBUG
      int m = *(Os.end() - 1);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  }
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::cumulativeOffspring(Random& rng, const V1 lws,
    V2 Os, const int n, int ka, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /// @todo May only work if ka == 0

  /* pre-condition */
  BI_ASSERT(lws.size() == Os.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type vector_type;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();

  if (sort) {
    int_vector_type os(P);
    offspring(rng, lws, os, n, ka, sorted, lws1, ps, Ws);
    sum_inclusive_scan(os, Os);
  } else {
    sumexpu_inclusive_scan(lws, Ws);
    T1 W = *(Ws.end() - 1);  // sum of weights

    if (W > 0) {
      int k = bi::find(ps, ka);
      T1 left = k > 0 ? *(Ws.begin() + k - 1) : 0.0;
      T1 right = *(Ws.begin() + k);
      real c = rng.uniform(left, right);
      int strata = std::floor(n * c / W);
      T1 a = n * c / W - strata;

      op_elements(Ws, Os, resample_cumulative_offspring<T1>(a, W, n));

#ifndef NDEBUG
      int m = *(Os.end() - 1);
      BI_ASSERT_MSG(m == n,
          "Systematic resampler gives " << m << " offspring, should give " << n);
#endif
    } else {
      throw ParticleFilterDegeneratedException();
    }
  }
}

template<class V1, class V2>
void bi::SystematicResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == lws.size());

  const int P = as.size();

  typename sim_temp_vector<V2>::type Os(P), ps(P);
  typename sim_temp_vector<V1>::type lws1(P), Ws(P);

  cumulativeOffspring(rng, lws, Os, P, false, lws1, ps, Ws);
  cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::ancestors(Random& rng, const V1 lws, V2 as,
    int P, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typename sim_temp_vector<V2>::type Os(lws.size());

  cumulativeOffspring(rng, lws, Os, P, sorted, lws1, ps, Ws);
  cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2, class V3, class V4>
void bi::SystematicResampler::ancestors(Random& rng, const V1 lws, V2 as,
    int P, int ka, int k, bool sorted, V3 lws1, V4 ps, V3 Ws)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typename sim_temp_vector<V2>::type Os(lws.size());

  cumulativeOffspring(rng, lws, Os, P, ka, sorted, lws1, ps, Ws);
  cumulativeOffspringToAncestors(Os, as);

  /* post-condition */
  BI_ASSERT(*(as.begin() + k) == ka);
}

#endif
