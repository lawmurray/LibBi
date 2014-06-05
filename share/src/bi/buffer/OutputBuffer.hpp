/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_OUTPUTBUFFER_HPP
#define BI_BUFFER_OUTPUTBUFFER_HPP

namespace bi {
/**
 * Output buffer. Can be used as concrete type when no output is required.
 *
 * @ingroup io_buffer
 */
class OutputBuffer {
public:
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
  void write(const size_t k, const real t, const S1& s) {
    //
  }

  /**
   * Write static components of state before simulation.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   */
  template<class S1>
  void write0(const S1& s) {
    //
  }

  /**
   * Write static components of state after simulation.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   */
  template<class S1>
  void writeT(const S1& s) {
    //
  }
};
}

#endif
