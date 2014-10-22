/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_INPUTNULLBUFFER_HPP
#define BI_NULL_INPUTNULLBUFFER_HPP

#include "../model/Model.hpp"
#include "../buffer/buffer.hpp"
#include "../math/scalar.hpp"
#include "../state/Mask.hpp"

namespace bi {
/**
 * Null input buffer.
 *
 * @ingroup io_null
 */
class InputNullBuffer {
public:
  /**
   * @copydoc InputNetCDFBuffer::InputNetCDFBuffer()
   */
  InputNullBuffer(const Model& m, const std::string& file = "",
      const long ns = 0, const long np = -1);

  /**
   * @copydoc InputNetCDFBuffer::getTime()
   */
  real getTime(const size_t k);

  /**
   * @copydoc InputBuffer::readTimes()
   */
  template<class T1>
  void readTimes(std::vector<T1>& ts);

  /**
   * @copydoc InputBuffer::readMask()
   */
  void readMask(const size_t k, const VarType type, Mask<ON_HOST>& mask);

  /**
   * @copydoc InputBuffer::read()
   */
  template<class M1>
  void read(const size_t k, const VarType type, const Mask<ON_HOST>& mask,
      M1 X);

  /**
   * @copydoc InputBuffer::read()
   */
  template<class M1>
  void read(const size_t k, const VarType type, M1 X);

  /**
   * @copydoc InputBuffer::readMask0()
   */
  void readMask0(const VarType type, Mask<ON_HOST>& mask);

  /**
   * @copydoc InputBuffer::read0()
   */
  template<class M1>
  void read0(const VarType type, const Mask<ON_HOST>& mask, M1 X);

  /**
   * @copydoc InputBuffer::read0()
   */
  template<class M1>
  void read0(const VarType type, M1 X);
};
}

inline real bi::InputNullBuffer::getTime(const size_t k) {
  BI_ERROR_MSG(false, "time index outside valid range");
}

template<class T1>
inline void bi::InputNullBuffer::readTimes(std::vector<T1>& ts) {
  ts.clear();
}

template<class M1>
void bi::InputNullBuffer::read(const size_t k, const VarType type,
    const Mask<ON_HOST>& mask, M1 X) {
  BI_ERROR_MSG(false, "time index outside valid range");
}

template<class M1>
void bi::InputNullBuffer::read(const size_t k, const VarType type, M1 X) {
  BI_ERROR_MSG(false, "time index outside valid range");
}

template<class M1>
void bi::InputNullBuffer::read0(const VarType type, const Mask<ON_HOST>& mask,
    M1 X) {
  //
}

template<class M1>
void bi::InputNullBuffer::read0(const VarType type, M1 X) {
  //
}

#endif
