/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_NETCDFBUFFER_HPP
#define BI_BUFFER_NETCDFBUFFER_HPP

#include "../model/BayesNet.hpp"
#include "../method/misc.hpp"

#include "netcdfcpp.h"

/**
 * NetCDF type identifier for real.
 *
 * @ingroup io
 */
extern NcType netcdf_real;

namespace bi {
/**
 * Result buffer supported by NetCDF file.
 *
 * @ingroup io
 */
class NetCDFBuffer {
public:
  /**
   * File open flags.
   */
  enum FileMode {
    /**
     * Open file read-only.
     */
    READ_ONLY,

    /**
     * Open file for reading and writing,
     */
    WRITE,

    /**
     * Open file for reading and writing, replacing any existing file of the
     * same name.
     */
    REPLACE,

    /**
     * Open file for reading and writing, fails if any existing file of the
     * same name
     */
    NEW

  };

  /**
   * Does file have given dimension?
   *
   * @param name Name of dimension.
   */
  bool hasDim(const char* name);

  /**
   * Does file have given variable?
   *
   * @param name Name of variable.
   */
  bool hasVar(const char* name);

  /**
   * Synchronize with file system.
   */
  void sync();

protected:
  /**
   * Constructor.
   *
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  NetCDFBuffer(const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Destructor.
   */
  virtual ~NetCDFBuffer();

  /**
   * Create dimension in NetCDF file.
   *
   * @param name Name.
   * @param size Size.
   *
   * @return The dimension.
   */
  NcDim* createDim(const char* name, const long size);

  /**
   * Create variable in NetCDF file.
   *
   * @param node Node in model for which to create variable in NetCDF file.
   * @param SH Handling of p-nodes and s-nodes for output.
   *
   * @return The variable.
   */
  NcVar* createVar(const BayesNode* node, const StaticHandling SH = STATIC_OWN);

  /**
   * Map dimension in existing NetCDF file.
   *
   * @param name Name of dimension.
   * @param size Expected size of dimension. Used to validate file, ignored
   * if negative.
   *
   * @return The dimension.
   */
  NcDim* mapDim(const char* name, const long size = -1);

  /**
   * Map variable in existing NetCDF file.
   *
   * @param node Node in model for which to map variable in NetCDF file.
   * @param SH Handling of p-nodes and s-nodes for output.
   *
   * @return The variable.
   */
  NcVar* mapVar(const BayesNode* node, const StaticHandling SH = STATIC_OWN);

  /**
   * NetCDF file.
   */
  NcFile* ncFile;
};
}

#include "../misc/assert.hpp"

inline bool bi::NetCDFBuffer::hasDim(const char* name) {
  /* pre-condition */
  assert (name != NULL);

  NcDim* dim = ncFile->get_dim(name);

  return (dim != NULL && dim->is_valid());
}

inline bool bi::NetCDFBuffer::hasVar(const char* name) {
  /* pre-condition */
  assert (name != NULL);

  NcVar* var = ncFile->get_var(name);

  return (var != NULL && var->is_valid());
}

inline void bi::NetCDFBuffer::sync() {
  ncFile->sync();
}

#endif
