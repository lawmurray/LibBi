/**
 * @file
 *
 * @author lee27x
 * $Rev$
 * $Date$
 */

#ifndef VARSTOPPER_HPP_
#define VARSTOPPER_HPP_

namespace bi {

class VarStopper {
public:
  VarStopper(int rel_threshold, int maxParticles);

  template<class V1> bool stop(V1 lws, int T, real maxlw, int blockSize);
  template<class V1> bool stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize);

  int getMaxParticles();

private:
  const int rel_threshold;
  const int maxParticles;
  real sum;
};

bi::VarStopper::VarStopper(int rel_threshold, int maxParticles) :
    rel_threshold(rel_threshold),
    maxParticles(maxParticles) {

}

template<class V1>
bool bi::VarStopper::stop(V1 lws, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
//  assert (start == 0 ? sumw == sumw2 && sumw == 0 : true);
  int start;
  if (lws.size() == blockSize) {
    start = 0;
    sum = 0;
  } else {
    start = lws.size() - blockSize;
  }

  real mu = sumexp_reduce(subrange(lws,start,blockSize))/blockSize;
  real s2 = sumexpsq_reduce(subrange(lws,start,blockSize))/blockSize;
  real val = s2-mu*mu;

  sum += blockSize*mu/val;

  real threshold = T*rel_threshold;

  assert (max_reduce(subrange(lws,start,lws.size()-start)) <= maxlw );

  if (lws.size() >= maxParticles) {
    return true;
  }
  if (sum >= threshold) {
    return true;
  } else {
    return false;
  }
}

template<class V1>
bool bi::VarStopper::stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
  int start;
  if (lws_1.size() == blockSize) {
    start = 0;
    sum = 0;
  } else {
    start = lws_1.size() - blockSize;
  }

  real mu = (sumexp_reduce(subrange(lws_1,start,blockSize))
          + sumexp_reduce(subrange(lws_2,start,blockSize)))/(2*blockSize);
  real s2 = (sumexpsq_reduce(subrange(lws_1,start,blockSize))
      + sumexpsq_reduce(subrange(lws_2,start,blockSize)))/(2*blockSize);
  real val = s2-mu*mu;

  sum += blockSize*mu/val;

  real threshold = 2*T*rel_threshold;

  if (lws_1.size() >= maxParticles) {
    return true;
  }

  if (sum >= threshold) {
    return true;
  } else {
    return false;
  }

}

int bi::VarStopper::getMaxParticles() {
  return maxParticles;
}

}


#endif
