/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SRSBUFFER_HPP
#define BI_BUFFER_SRSBUFFER_HPP

#include "MCMCBuffer.hpp"

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of marginal SRS.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class SRSBuffer: public MCMCBuffer<IO1> {
public:
  typedef MCMCBuffer<IO1> parent_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file File name.
   * @param mode File open mode.
   */
  SRSBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

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
bi::SRSBuffer<IO1>::SRSBuffer(const Model& m, const size_t P, const size_t T,
    const std::string& file, const FileMode mode, const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema) {
  //
}

template<class IO1>
template<class S1>
void bi::SRSBuffer<IO1>::write(const int c, const S1& s) {
  parent_type::write(c, s);
  parent_type::writeLogWeight(c, s.logPrior + s.logLikelihood - s.logProposal);
}

#endif
