/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_MULTINOMIALRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_MULTINOMIALRESAMPLERHOST_HPP

template<class V1, class V2, class V3, class V4>
void bi::MultinomialResamplerHost::ancestors(Random& rng, const V1 lws, V2 as,
    int P, bool sort, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typedef typename V1::value_type T1;
  const int lwsSize = lws.size();

  typename sim_temp_vector<V1>::type alphas(P);
  typename sim_temp_vector<V2>::type as1(P);
  T1 W, lZ;

  /* weights */
  if (sort) {
    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      lZ = bi::exclusive_scan_sum_expu(lws1, Ws);
    }
  } else {
    lZ = bi::exclusive_scan_sum_expu(lws, Ws);
  }
  W = *(Ws.end() - 1) + bi::exp(*(lws.end() - 1) - lZ); // sum of weights
  if (W > 0) {
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      int Q = P/nthreads;
      int start = tid*Q + bi::min(tid, P % nthreads);
      if (tid < P % nthreads) {
        ++Q; // pick up a leftover
      }

      int i, j = lwsSize - 1;
      T1 lMax = 0.0, u;
      for (i = Q; i > 0; --i) {
        lMax += bi::log(rng.uniform<T1>())/i;
        u = W*bi::exp(lMax);

        while (u < Ws(j)) {
          --j;
        }
        if (sort) {
          as1(start + i - 1) = j;
        } else {
          as(start + i - 1) = j;
        }
      }
    }

    if (sort) {
      thrust::gather(as1.begin(), as1.end(), ps.begin(), as.begin());
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }

  /* post-condition */
  BI_ASSERT(max_reduce(as) < lws.size());
}

#endif