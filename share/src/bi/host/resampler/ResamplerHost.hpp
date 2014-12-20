/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_RESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_RESAMPLERHOST_HPP

namespace bi {
/**
 * Resampler implementation on host.
 */
class ResamplerHost {
public:
  /**
   * @copydoc Resampler::ancestorsToOffspring()
   */
  template<class V1, class V2>
  static void ancestorsToOffspring(const V1 as, V2 os);

  /**
   * @copydoc Resampler::offspringToAncestors()
   */
  template<class V1, class V2>
  static void offspringToAncestors(const V1 os, V2 as);

  /**
   * @copydoc Resampler::offspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void offspringToAncestorsPermute(const V1 os, V2 as);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestors()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestors(const V1 Os, V2 as);

  /**
   * @copydoc Resampler::cumulativeOffspringToAncestorsPermute()
   */
  template<class V1, class V2>
  static void cumulativeOffspringToAncestorsPermute(const V1 Os, V2 as);

  /**
   * @copydoc Resampler::permute()
   */
  template<class V1>
  static void permute(V1 as);
};
}

#include "../../primitive/vector_primitive.hpp"

template<class V1, class V2>
void bi::ResamplerHost::ancestorsToOffspring(const V1 as, V2 os) {
  /* pre-conditions */
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!V2::on_device);

  os.clear();
  for (int p = 0; p < as.size(); ++p) {
    ++os(as(p));
  }

  /* post-condition */
  BI_ASSERT(sum_reduce(os) == as.size());
}

template<class V1, class V2>
void bi::ResamplerHost::offspringToAncestors(const V1 os, V2 as) {
  /* pre-conditions */
  BI_ASSERT(sum_reduce(os) == as.size());
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!V2::on_device);

  int i, j, k = 0, o;
  for (i = 0; i < os.size(); ++i) {
    o = os(i);
    for (j = 0; j < o; ++j) {
      as(k++) = i;
    }
  }
}

template<class V1, class V2>
void bi::ResamplerHost::offspringToAncestorsPermute(const V1 os, V2 as) {
  /* pre-conditions */
  BI_ASSERT(sum_reduce(os) == as.size());
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!V2::on_device);

  int i, j, k = 0, o;
  for (i = 0; i < os.size(); ++i) {
    o = os(i);
    if (o > 0) {
      as(i) = i;
      --o;
    }
    for (j = 0; j < o; ++j) {
      while (os(k) > 0) {
        ++k;
      }
      as(k++) = i;
    }
  }
}

template<class V1, class V2>
void bi::ResamplerHost::cumulativeOffspringToAncestors(const V1 Os, V2 as) {
  /* pre-conditions */
  BI_ASSERT(*(Os.end() - 1) == as.size());
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!V2::on_device);

  #pragma omp parallel
  {
    int i, j, O1, O2, o;

    #pragma omp for
    for (int i = 0; i < Os.size(); ++i) {
      O1 = (i > 0) ? Os(i - 1) : 0;
      O2 = Os(i);
      o = O2 - O1;

      for (j = 0; j < o; ++j) {
        as(O1 + j) = i;
      }
    }
  }
}

template<class V1, class V2>
void bi::ResamplerHost::cumulativeOffspringToAncestorsPermute(const V1 Os,
    V2 as) {
  /* pre-conditions */
  BI_ASSERT(*(Os.end() - 1) == as.size());
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!V2::on_device);

  int i, j, k = 0, o, O1, O2;
  for (i = 0; i < Os.size(); ++i) {
    O1 = (i > 0) ? Os(i - 1) : 0;
    O2 = Os(i);
    o = O2 - O1;

    if (o > 0) {
      as(i) = i;
      --o;
    }

    if (k == 0 && o > 0) { // deal with special case here rather than in loop
      if (Os(k) > 0) {
        ++k;
      } else {
        as(k++) = i;
        --o;
      }
    }

    for (j = 0; j < o; ++j) {
      while (Os(k) - Os(k - 1) > 0) {
        ++k;
      }
      as(k++) = i;
    }
  }
}

template<class V1>
void bi::ResamplerHost::permute(V1 as) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);

  const int P = as.size();

  typename V1::size_type i;
  typename V1::value_type j, k;

  for (i = 0; i < as.size(); ++i) {
    k = as(i);
    if (k < P && k != i && as(k) != k) {
      /* swap */
      j = as(k);
      as(k) = k;
      as(i) = j;
      --i; // repeat for new value
    }
  }
}

#endif
