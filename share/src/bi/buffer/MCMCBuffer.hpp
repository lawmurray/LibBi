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
   * @param file File name.
   * @param mode File open mode.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   */
  MCMCBuffer(const Model& m, const std::string& file, const FileMode mode =
      READ_ONLY, const SchemaMode schema = DEFAULT, const size_t P = 0,
      const size_t T = 0);

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
bi::MCMCBuffer<IO1>::MCMCBuffer(const Model& m, const std::string& file,
    const FileMode mode, const SchemaMode schema, const size_t P,
    const size_t T) :
    IO1(m, file, mode, schema, P, T) {
  //
}

template<class IO1>
template<class S1>
void bi::MCMCBuffer<IO1>::write(const int c, const S1& s) {
  if (c == 0) {
    //IO1::writeTimes(0, s.getTimes());
  }
  IO1::writeLogLikelihood(c, s.logLikelihood1);
  IO1::writeLogPrior(c, s.logPrior1);
  IO1::writeParameter(c, s.theta1);
  IO1::writePath(c, s.path);
}

#endif
