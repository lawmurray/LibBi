/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_STRATIFIEDRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_STRATIFIEDRESAMPLERHOST_HPP

#include "ResamplerHost.hpp"

namespace bi {
/**
 * StratifiedResampler implementation on host.
 */
class StratifiedResamplerHost: public ResamplerHost {
public:
  /**
   * @copydoc StratifiedResampler::op
   */
  template<class V1, class V2>
  static void op(Random& rng, const V1 Ws, V2 Os, const int n);
};
}

template<class V1, class V2>
void bi::StratifiedResamplerHost::op(Random& rng, const V1 Ws, V2 Os,
    const int n) {
  /* pre-condition */
  BI_ASSERT(Ws.size() == Os.size());

  typedef typename V1::value_type T1;
  host_vector<T1> alphas(n);

  const int P = Ws.size();
  const T1 W = *(Ws.end() - 1);

  /*
   * Under the Intel compiler (version 12.1.3), a segfault is triggered on the
   * next line, specifically when entering the <tt>#pragma omp parallel
   * for</tt> region internal to
   * <tt>StratifiedResamplerHost::uniforms()</tt>. This seems to be a compiler
   * bug, and the implementation here is a workaround. The contents of that
   * function are copied into this one, and the two <tt>#pragma omp
   * parallel</tt> blocks combined. This now seems to work.
   */
  //rng.uniforms(alphas);

  typedef typename V1::value_type T1;
  typedef boost::uniform_real<T1> dist_type;

  #pragma omp parallel
  {
    RngHost& rng1 = rng.getHostRng();
    int i;

    dist_type dist(0.0, 1.0);
    boost::variate_generator<RngHost::rng_type&, dist_type> gen(rng1.rng, dist);

    #pragma omp for
    for (i = 0; i < alphas.size(); ++i) {
      alphas(i) = gen();
    }

    #pragma omp barrier

    #pragma omp for
    for (i = 0; i < Ws.size(); ++i) {
      T1 reach = Ws(i)/W*n;
      int k = bi::min(n - 1, static_cast<int>(reach));

      Os(i) = bi::min(n, static_cast<int>(reach + alphas(k)));
    }
  }
}

#endif
