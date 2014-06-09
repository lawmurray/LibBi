/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SMCBUFFER_HPP
#define BI_BUFFER_SMCBUFFER_HPP

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of marginal MH.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class SMCBuffer: public IO1 {
public:
  /**
   * Pass-through constructor.
   */
  SMCBuffer();

  /**
   * Pass-through constructor.
   */
  template<class T1>
  SMCBuffer(T1& o1);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  SMCBuffer(T1& o1, T2& o2);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  SMCBuffer(T1& o1, T2& o2, T3& o3);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  SMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  SMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5);

  /**
   * Write sample.
   *
   * @tparam S1 State type.
   *
   * @param c Sample index.
   * @param s State.
   */
  template<class S1>
  void write(const S1& s);
};
}

template<class IO1>
bi::SMCBuffer<IO1>::SMCBuffer() {
  //
}

template<class IO1>
template<class T1>
bi::SMCBuffer<IO1>::SMCBuffer(T1& o1) :
    IO1(o1) {
  //
}

template<class IO1>
template<class T1, class T2>
bi::SMCBuffer<IO1>::SMCBuffer(T1& o1, T2& o2) :
    IO1(o1, o2) {
  //
}

template<class IO1>
template<class T1, class T2, class T3>
bi::SMCBuffer<IO1>::SMCBuffer(T1& o1, T2& o2, T3& o3) :
    IO1(o1, o2, o3) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4>
bi::SMCBuffer<IO1>::SMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4) :
    IO1(o1, o2, o3, o4) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4, class T5>
bi::SMCBuffer<IO1>::SMCBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5) :
    IO1(o1, o2, o3, o4, o5) {
  //
}

template<class IO1>
template<class S1>
void bi::SMCBuffer<IO1>::write(const S1& s) {
  //IO1::write(c, s);
  writeLogWeights(s.lws);
  writeLogEvidences(s.les);
}

#endif
