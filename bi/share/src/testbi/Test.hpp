/**
 * @file
 *
 * @author lee27x
 * $Rev$
 * $Date$
 */

#ifndef TEST_HPP_
#define TEST_HPP_

#include "bi/random/Random.hpp"

namespace bi {

class Test {

public:

  inline static bool test(Random& rng);

};

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
}

#include "thrust/adjacent_difference.h"

inline bool bi::Test::test(Random& rng) {
  std::cerr << "Hello World!" << std::endl;
  bool sort = true;

  int P = 10, ka = 3;
  int n = 100;
  typename temp_host_vector<real>::type vlws1(P), vWs(P);
  typename temp_host_vector<int>::type vas(P), vps(P), vOs(P), vos(P);
  typename temp_host_vector<real>::type lws1 = vlws1.ref();
  typename temp_host_vector<real>::type Ws = vWs.ref();
  typename temp_host_vector<int>::type as = vas.ref();
  typename temp_host_vector<int>::type ps = vps.ref();
  typename temp_host_vector<int>::type Os = vOs.ref();
  typename temp_host_vector<int>::type os = vos.ref();

  seq_elements(as, 0);
  rng.uniforms(lws1,0.0,1.0);

//  std::cerr << "initial: ";
//  for (int i = 0; i < P; i++) {
//    std::cerr << bi::exp(lws1[i]) << " ";
//  }
//  std::cerr << std::endl;

  if (sort) {
    seq_elements(ps, 0);
    bi::sort_by_key(lws1, ps);
  }

//  std::cerr << "sorted: ";
//  for (int i = 0; i < P; i++) {
//    std::cerr << bi::exp(lws1[i]) << " ";
//  }
//  std::cerr << std::endl;

  bi::inclusive_scan_sum_expu(lws1, Ws);
  real W = *(Ws.end() - 1); // sum of weights

//  std::cerr << "cumulative: ";
//  for (int i = 0; i < P; i++) {
//    std::cerr << Ws[i] << " ";
//  }
//  std::cerr << std::endl;


  int k = bi::find(ps,ka);
//  std::cerr << "k = " << k << std::endl;

  real left = k > 0 ? Ws[k-1] : 0;
  real right = Ws[k];
  std::cerr << "E[o_k] = " << (right-left)/W*n << std::endl;
  real c = rng.uniform(left,right);
//  std::cerr << "c = " << c << " ~ U(" << left << "," << right << ")" << std::endl;
  int strata = std::floor(n*c/W);
  real a = n*c/W - strata;
//  std::cerr << "strata = " << strata << ". a = " << a << std::endl;
//  std::cerr << "values: ";
//  for (int i = 0; i < n; i++) {
//    std::cerr << i*W/n + a << " ";
//  }
//  std::cerr << std::endl;

  if (W > 0) {
  //    a = rng.uniform(0.0, 1.0); // offset into strata
    thrust::transform(Ws.begin(), Ws.end(), Os.begin(), resample_offspring<real>(a, W, n));
    thrust::adjacent_difference(Os.begin(), Os.end(), os.begin());

//    for (int i = 0; i < P; i++) {
//      std::cerr << os[i] << " ";
//    }
//    std::cerr << std::endl;

    if (sort) {
      typename temp_host_vector<int>::type vtemp(P);
      typename temp_host_vector<int>::type temp = vtemp.ref();
      temp = os;
      bi::scatter(ps,temp,os);
    }
  }

//  for (int i = 0; i < P; i++) {
//    std::cerr << os[i] << " ";
//  }
//  std::cerr << std::endl;

  std::cerr << "os[ka] = " << os[ka] << std::endl;








  return true;
}


#endif
