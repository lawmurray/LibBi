/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_INPUTBUFFER_HPP
#define BI_BUFFER_INPUTBUFFER_HPP

#include "buffer.hpp"
#include "../model/Model.hpp"
#include "../state/Mask.hpp"

#include <vector>

namespace bi {
/**
 * Input buffer. Can be used as concrete type when no input is required.
 *
 * @ingroup io_buffer
 */
class InputBuffer {
public:
  /**
   * Get current time.
   */
  real getTime(const size_t k) {
    return 0.0;
  }

  /**
   * Read times.
   *
   * @tparam T1 Scalar type.
   *
   * @param ts Times.
   *
   * The times are in ascending order, without duplicates.
   */
  template<class T1>
  void readTimes(std::vector<T1>& ts) {
    //
  }

  /**
   * Read mask of dynamic variables.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param[out] mask Mask.
   */
  void readMask(const size_t k, const VarType type, Mask<ON_HOST>& mask) {
    //
  }

  /**
   * Read dynamic variables.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param mask Mask.
   * @param[in,out] X State.
   */
  template<class M1>
  void read(const size_t k, const VarType type, const Mask<ON_HOST>& mask,
      M1 X) {
    //
  }

  /**
   * Read dynamic variables.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param[in,out] X State.
   */
  template<class M1>
  void read(const size_t k, const VarType type, M1 X) {
    //
  }

  /**
   * Read mask of static variables.
   *
   * @param type Variable type.
   * @param[out] mask Mask.
   */
  void readMask0(const VarType type, Mask<ON_HOST>& mask) {
    //
  }

  /**
   * Read static variables.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param mask Mask.
   * @param[in,out] X State.
   */
  template<class M1>
  void read0(const VarType type, const Mask<ON_HOST>& mask, M1 X) {
    //
  }

  /**
   * Convenience method for reading static variables when the mask is not of
   * interest.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param[in,out] X State.
   */
  template<class M1>
  void read0(const VarType type, M1 X) {
    //
  }
};
}

#endif
