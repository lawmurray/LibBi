/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_STRATIFIEDRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_STRATIFIEDRESAMPLERHOST_HPP

template<class V1, class V2>
void bi::StratifiedResamplerHost::op(Random& rng, const V1 Ws, V2 Os,
    const int n) {
  /* pre-condition */
  BI_ASSERT(Ws.size() == Os.size());

  typedef typename V1::value_type T1;
  typename sim_temp_vector<V1>::type alphas(n);

  const int P = Ws.size();
  const T1 W = *(Ws.end() - 1);

  rng.uniforms(alphas);

  #pragma omp parallel for
  for (int i = 0; i < P; ++i) {
    T1 reach = Ws(i)/W*n;
    int k = bi::min(n - 1, static_cast<int>(reach));

    Os(i) = static_cast<int>(reach + alphas(k));
  }
}

#endif
