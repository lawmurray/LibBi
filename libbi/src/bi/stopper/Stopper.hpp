/**
 * @file
 *
 * @author anthony
 * $Rev$
 * $Date$
 */

#ifndef STOPPER_HPP_
#define STOPPER_HPP_

#include "../math/scalar.hpp"

namespace bi {

class Stopper {
public:
  Stopper(int P);

  template<class V1> bool stop(V1 lws, int T, real maxlw, int blockSize);
  template<class V1> bool stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize);

  int getMaxParticles();

private:
  const int P;
};

bi::Stopper::Stopper(int P) : P(P) {

}

template<class V1>
bool bi::Stopper::stop(V1 lws, int T, real maxlw, int blockSize) {
  return lws.size() >= P;
}

template<class V1>
bool bi::Stopper::stop(V1 lws_1, V1 lws_2, int T, real maxlw, int blockSize) {
  return lws_1.size() >= P;
}

int bi::Stopper::getMaxParticles() {
  return P;
}

}


#endif
