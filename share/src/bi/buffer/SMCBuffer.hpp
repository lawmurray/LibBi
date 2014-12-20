/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SMCBUFFER_HPP
#define BI_BUFFER_SMCBUFFER_HPP

#include "MCMCBuffer.hpp"

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of marginal MH.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class SMCBuffer: public MCMCBuffer<IO1> {
public:
  typedef MCMCBuffer<IO1> parent_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param file File name.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param mode File open mode.
   */
  SMCBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * Write sample.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   */
  template<class S1>
  void write(const S1& s);
};
}

template<class IO1>
bi::SMCBuffer<IO1>::SMCBuffer(const Model& m, const size_t P, const size_t T,
    const std::string& file, const FileMode mode, const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema) {
  //
}

template<class IO1>
template<class S1>
void bi::SMCBuffer<IO1>::write(const S1& s) {
  for (int p = 0; p < s.size(); ++p) {
    parent_type::write(p, *s.s1s[p]);
    if (this->isFull()) {
      this->flush();
      this->clear();
    }
  }
  parent_type::writeLogWeights(0, s.logWeights());
}

#endif
