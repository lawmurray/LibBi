/**
 * @file
 *
 * @author anthony
 * $Rev$
 * $Date$
 */

#ifndef SUMOFWEIGHTSSTOPPER_HPP_
#define SUMOFWEIGHTSSTOPPER_HPP_

namespace bi {

class SumOfWeightsStopper {
public:
  SumOfWeightsStopper(int rel_threshold, int maxParticles);

  template<class V1> bool stop(V1 lws, int T, real maxlw, int blockSize);
  template<class V1> bool stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize);

  int getMaxParticles();

private:
  const int rel_threshold;
  const int maxParticles;
  real sumw;
};

bi::SumOfWeightsStopper::SumOfWeightsStopper(int rel_threshold, int maxParticles) :
    rel_threshold(rel_threshold),
    maxParticles(maxParticles) {

}

template<class V1>
bool bi::SumOfWeightsStopper::stop(V1 lws, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
  int start;
  if (lws.size() == blockSize) {
    start = 0;
    sumw = 0;
  } else {
    start = lws.size() - blockSize;
  }
  sumw += sumexp_reduce(subrange(lws,start,blockSize));

  real threshold = T*rel_threshold*exp(maxlw);

  assert (max_reduce(subrange(lws,start,blockSize)) <= maxlw );

  if (lws.size() >= maxParticles) {
    return true;
  }

  if (sumw >= threshold) {
    return true;
  } else {
    return false;
  }
}

template<class V1>
bool bi::SumOfWeightsStopper::stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
  int start;
  if (lws_1.size() == blockSize) {
    start = 0;
    sumw = 0;
  } else {
    start = lws_1.size() - blockSize;
  }
  sumw += sumexp_reduce(subrange(lws_1,start,blockSize)) + sumexp_reduce(subrange(lws_2,start,blockSize));

  real threshold = 2*T*rel_threshold*exp(maxlw);

  assert (max_reduce(subrange(lws_1,start,blockSize)) <= maxlw);
  assert (max_reduce(subrange(lws_2,start,blockSize)) <= maxlw);

  if (lws_1.size() >= maxParticles) {
    return true;
  }

  if (sumw >= threshold) {
    return true;
  } else {
    return false;
  }
}

int bi::SumOfWeightsStopper::getMaxParticles() {
  return maxParticles;
}

}


#endif
