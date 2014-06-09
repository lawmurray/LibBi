/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEFILTERBUFFER_HPP
#define BI_BUFFER_PARTICLEFILTERBUFFER_HPP

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of a filter.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class ParticleFilterBuffer: public IO1 {
public:
  /**
   * Pass-through constructor.
   */
  ParticleFilterBuffer();

  /**
   * Pass-through constructor.
   */
  template<class T1>
  ParticleFilterBuffer(T1& o1);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  ParticleFilterBuffer(T1& o1, T2& o2);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  ParticleFilterBuffer(T1& o1, T2& o2, T3& o3);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  ParticleFilterBuffer(T1& o1, T2& o2, T3& o3, T4& o4);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  ParticleFilterBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5);

  /**
   * Write state.
   *
   * @tparam S1 State type.
   *
   * @param k Time index.
   * @param t Time.
   * @param s State.
   */
  template<class S1>
  void write(const size_t k, const real t, const S1& s);
};
}

template<class IO1>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer() {
  //
}

template<class IO1>
template<class T1>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(T1& o1) :
    IO1(o1) {
  //
}

template<class IO1>
template<class T1, class T2>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(T1& o1, T2& o2) :
    IO1(o1, o2) {
  //
}

template<class IO1>
template<class T1, class T2, class T3>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(T1& o1, T2& o2, T3& o3) :
    IO1(o1, o2, o3) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(T1& o1, T2& o2, T3& o3,
    T4& o4) :
    IO1(o1, o2, o3, o4) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4, class T5>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(T1& o1, T2& o2, T3& o3,
    T4& o4, T5& o5) :
    IO1(o1, o2, o3, o4, o5) {
  //
}

template<class IO1>
template<class S1>
void bi::ParticleFilterBuffer<IO1>::write(const size_t k, const real t,
    const S1& s) {
  IO1::writeTime(k, t);
  IO1::writeState(k, s.getDyn(), s.as);
  IO1::writeLogWeights(k, s.lws);
}

#endif
