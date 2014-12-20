/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_MCMCBUFFER_HPP
#define BI_BUFFER_MCMCBUFFER_HPP

#include "buffer.hpp"

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
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file File name.
   * @param mode File open mode.
   * @param scheme File schema.
   */
  MCMCBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
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
bi::MCMCBuffer<IO1>::MCMCBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    IO1(m, P, T, file, mode, schema) {
  //
}

template<class IO1>
template<class S1>
void bi::MCMCBuffer<IO1>::write(const int c, const S1& s) {
  IO1::writeLogLikelihood(c, s.logLikelihood);
  IO1::writeLogPrior(c, s.logPrior);
  IO1::writeParameter(c, row(s.get(P_VAR), 0));
  if (c == 0) {
    IO1::writeTimes(0, s.times);
  }
  IO1::writePath(c, s.path);
}

#endif
