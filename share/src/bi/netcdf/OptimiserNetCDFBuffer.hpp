/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_OPTIMISERNETCDFBUFFER_HPP
#define BI_NETCDF_OPTIMISERNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"
#include "../state/State.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of
 * NelderMeadOptimiser.
 *
 * @ingroup io_netcdf
 */
class OptimiserNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param T Number of times to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  OptimiserNetCDFBuffer(const Model& m, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = PARAM_ONLY);

  /**
   * @copydoc concept::OptimiserBuffer::writeValue()
   */
  void writeValue(const size_t k, const real& x);

  /**
   * @copydoc concept::OptimiserBuffer::writeSize()
   */
  void writeSize(const size_t k, const real& x);

protected:
  /**
   * Set up structure of NetCDF file.
   */
  void create();

  /**
   * Map structure of existing NetCDF file.
   */
  void map();

  /**
   * Function value variable.
   */
  int valueVar;

  /**
   * Size variable.
   */
  int sizeVar;
};
}

#endif
