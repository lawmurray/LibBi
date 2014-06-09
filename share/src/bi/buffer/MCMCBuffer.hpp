/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_MCMCBUFFER_HPP
#define BI_BUFFER_MCMCBUFFER_HPP

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of marginal MH.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class MCMCBuffer: public IO1 {
public:
  /**
   * Pass-through constructor.
   */
  MCMCBuffer();

  /**
   * Pass-through constructor.
   */
  template<class T1>
  MCMCBuffer(T1& o1);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  MCMCBuffer(T1& o1, T2& o2);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  MCMCBuffer(T1& o1, T2& o2, T3& o3);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  MCMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  MCMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5);

  /**
   * Write sample.
   *
   * @tparam S1 State type.
   *
   * @param c Sample index.
   * @param s State.
   */
  template<class S1>
  void write(const int c, const S1& s);
};
}

template<class IO1>
bi::MCMCBuffer<IO1>::MCMCBuffer() {
  //
}

template<class IO1>
template<class T1>
bi::MCMCBuffer<IO1>::MCMCBuffer(T1& o1) :
    IO1(o1) {
  //
}

template<class IO1>
template<class T1, class T2>
bi::MCMCBuffer<IO1>::MCMCBuffer(T1& o1, T2& o2) :
    IO1(o1, o2) {
  //
}

template<class IO1>
template<class T1, class T2, class T3>
bi::MCMCBuffer<IO1>::MCMCBuffer(T1& o1, T2& o2, T3& o3) :
    IO1(o1, o2, o3) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4>
bi::MCMCBuffer<IO1>::MCMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4) :
    IO1(o1, o2, o3, o4) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4, class T5>
bi::MCMCBuffer<IO1>::MCMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5) :
    IO1(o1, o2, o3, o4, o5) {
  //
}

template<class IO1>
template<class S1>
void bi::MCMCBuffer<IO1>::write(const size_t k, const real t, const S1& s) {
  if (c == 0) {
    //IO1::writeTimes(0, s.getTimes());
  }
  IO1::writeLogLikelihood(c, s.logLikelihood1);
  IO1::writeLogPrior(c, s.logPrior1);
  IO1::writeParameter(c, s.theta1);
  IO1::writePath(c, s.path);
}

#endif
