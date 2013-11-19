/**
 * @file
 *
 * @author lee27x
 * $Rev$
 * $Date$
 */

#ifndef MINIMUMESSSTOPPER_HPP_
#define MINIMUMESSSTOPPER_HPP_

namespace bi {

class MinimumESSStopper {
public:
  MinimumESSStopper(int rel_min_ess, int maxParticles);

  template<class V1> bool stop(V1 lws, int T, real maxlw, int blockSize);
  template<class V1> bool stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize);

  int getMaxParticles();

private:
  const int rel_min_ess;
  const int maxParticles;
  real sumw;
  real sumw2;
};

bi::MinimumESSStopper::MinimumESSStopper(int rel_min_ess, int maxParticles) :
    rel_min_ess(rel_min_ess),
    maxParticles(maxParticles) {

}

template<class V1>
bool bi::MinimumESSStopper::stop(V1 lws, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
//  BI_ASSERT(start == 0 ? sumw == sumw2 && sumw == 0 : true);
  int start;
  if (lws.size() == blockSize) {
    start = 0;
    sumw = 0;
    sumw2 = 0;
  } else {
    start = lws.size() - blockSize;
  }
  sumw += sumexp_reduce(subrange(lws,start,blockSize));
  sumw2 += sumexpsq_reduce(subrange(lws,start,blockSize));
  T1 ess = (sumw*sumw)/sumw2;

  real min_ess = T*rel_min_ess;
  real threshold = exp(maxlw)*(min_ess-1)/2;
  BI_ASSERT(max_reduce(subrange(lws,start,blockSize)) <= maxlw );

  if (lws.size() >= maxParticles) {
    return true;
  }
  if (sumw >= threshold && ess >= min_ess) {
    return true;
  } else {
    return false;
  }
}

template<class V1>
bool bi::MinimumESSStopper::stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize) {
  typedef typename V1::value_type T1;
  int start;
  if (lws_1.size() == blockSize) {
    start = 0;
    sumw = 0;
    sumw2 = 0;
  } else {
    start = lws_1.size() - blockSize;
  }

  sumw += sumexp_reduce(subrange(lws_1,start,blockSize))
      + sumexp_reduce(subrange(lws_2,start,blockSize));
  sumw2 += sumexpsq_reduce(subrange(lws_1,start,blockSize))
      + sumexpsq_reduce(subrange(lws_2,start,blockSize));

  T1 ess = (sumw*sumw)/sumw2;

  real min_ess = 2*T*rel_min_ess;
  real threshold = exp(maxlw)*(min_ess-1)/2;

  if (lws_1.size() >= maxParticles) {
    return true;
  }

  if (sumw >= threshold && ess >= min_ess) {
    return true;
  } else {
    return false;
  }

}

int bi::MinimumESSStopper::getMaxParticles() {
  return maxParticles;
}

}


#endif
