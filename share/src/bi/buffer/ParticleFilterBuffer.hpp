/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEFILTERBUFFER_HPP
#define BI_BUFFER_PARTICLEFILTERBUFFER_HPP

#include "SimulatorBuffer.hpp"

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of a filter.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class ParticleFilterBuffer: public SimulatorBuffer<IO1> {
public:
  typedef SimulatorBuffer<IO1> parent_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param file File name.
   * @param mode File open mode.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   */
  ParticleFilterBuffer(const Model& m, const std::string& file = "",
      const FileMode mode = READ_ONLY, const SchemaMode schema = DEFAULT,
      const size_t P = 0, const size_t T = 0);

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

  /**
   * Write marginal log-likelihood
   *
   * @param ll Marginal log-likelihood.
   */
  void writeT(const real ll);
};
}

template<class IO1>
bi::ParticleFilterBuffer<IO1>::ParticleFilterBuffer(const Model& m,
    const std::string& file, const FileMode mode, const SchemaMode schema,
    const size_t P, const size_t T) :
    parent_type(m, file, mode, schema, P, T) {
  //
}

template<class IO1>
template<class S1>
void bi::ParticleFilterBuffer<IO1>::write(const size_t k, const real t,
    const S1& s) {
  parent_type::writeTime(k, t);
  parent_type::writeState(k, s.getDyn(), s.as);
  parent_type::writeLogWeights(k, s.lws);
}

template<class IO1>
void bi::ParticleFilterBuffer<IO1>::writeT(const real ll) {
  parent_type::writeLogLikelihood(ll);
}

#endif
