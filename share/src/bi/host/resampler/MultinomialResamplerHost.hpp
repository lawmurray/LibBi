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
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      int Q = P/nthreads;
      int start = tid*Q + bi::min(tid, P % nthreads); // min() handles leftovers
      if (tid < P % nthreads) {
        ++Q; // pick up a leftover
      }

      int i, j = lwsSize - 1;
      T1 lMax = 0.0, lu;
      for (i = Q; i > 0; --i) {
        lMax += bi::log(rng.uniform<T1>())/i;
        lu = lW + lMax;

        while (lu < bi::log(pre.Ws(j))) {
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
