/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_MULTINOMIALRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_MULTINOMIALRESAMPLERHOST_HPP

template<class V1, class V2>
void bi::MultinomialResamplerHost::ancestors(Random& rng, const V1 lws, V2 as,
    MultinomialPrecompute<ON_HOST>& pre)
    throw (ParticleFilterDegeneratedException) {
  typedef typename V1::value_type T1;

  const int P = as.size();
  const int lwsSize = lws.size();

  T1 lW;

  /* weights */
  if (pre.W > 0) {
    lW = bi::log(pre.W);

    #pragma omp parallel
    {
      int Q = P/bi_omp_max_threads;
      int start = bi_omp_tid*Q + bi::min(bi_omp_tid, P % bi_omp_max_threads); // min() handles leftovers
      if (bi_omp_tid < P % bi_omp_max_threads) {
        ++Q; // pick up a leftover
      }

      int i, j = lwsSize;
      T1 lMax = 0.0, lu;
      for (i = Q; i > 0; --i) {
        lMax += bi::log(rng.uniform<T1>())/i;
        lu = lW + lMax;

        while (j > 0 && lu < bi::log(pre.Ws(j - 1))) {
          --j;
        }
        if (pre.sort) {
          as(start + i - 1) = pre.ps(j);
        } else {
          as(start + i - 1) = j;
        }
      }
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }

  /* post-condition */
  BI_ASSERT(max_reduce(as) < lws.size());
}

#endif
