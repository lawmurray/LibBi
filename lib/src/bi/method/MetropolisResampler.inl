/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_METROPOLISRESAMPLER_INL
#define BI_METHOD_METROPOLISRESAMPLER_INL

template<class V1, class V2>
void bi::MetropolisResampler::resample(V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(lws, as);
  permute(as);
  lws.clear();
}

template<class V1, class V2>
void bi::MetropolisResampler::resample(const int a, V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(lws, as);
  as[0] = a;
  permute(as);
  lws.clear();
}

template<class V1, class V2, class V3>
void bi::MetropolisResampler::resample(const V1& qlws, V2& lws, V3& as) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(qlws, as);
  permute(as);
  correct(as, qlws, lws);
}

template<class V1, class V2, class V3>
void bi::MetropolisResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  ancestors(qlws, as);
  as[0] = a;
  permute(as);
  correct(as, qlws, lws);
}

template<class V1, class V2>
void bi::MetropolisResampler::ancestors(const V1& lws, V2& as) {
  typedef typename boost::mpl::if_c<V1::on_device,
      MetropolisResamplerDeviceImpl,
      MetropolisResamplerHostImpl>::type impl;
  impl::ancestors(lws, as, rng, L);
}

template<class V1, class V2>
void bi::MetropolisResamplerHostImpl::ancestors(const V1& lws, V2& as, Random& rng, int L) {
  const int P = lws.size();

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int l, p1, p2, tid;

    #pragma omp for
    for (tid = 0; tid < P; ++tid) {
      p1 = tid;
      lw1 = lws[tid];
      for (l = 0; l < L; ++l) {
        p2 = rng.uniformInt(0, P - 1);
        lw2 = lws[p2];
        alpha = rng.uniform<real>();

        if (alpha < CUDA_EXP(lw2 - lw1)) {
          /* accept */
          p1 = p2;
          lw1 = lw2;
        }
      }

      /* write result */
      as[tid] = p1;
    }
  }
}

#endif
