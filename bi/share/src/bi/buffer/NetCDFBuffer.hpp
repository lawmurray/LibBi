/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_NETCDFBUFFER_HPP
#define BI_BUFFER_NETCDFBUFFER_HPP

#include "../model/Model.hpp"
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
   * Copy constructor.
   *
   * Reopens the file of the argument with a new file handle, in read only
   * mode.
   */
  NetCDFBuffer(const NetCDFBuffer& o);

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
   * Create unlimited dimension in NetCDF file.
   *
   * @param name Name.
   *
   * @return The dimension.
   */
  NcDim* createDim(const char* name);

  /**
   * Create variable in NetCDF file.
   *
   * @param var Variable in model for which to create variable in NetCDF file.
   * @param nr Should @c nr dimension be forced, even for static variables?
   *
   * @return The variable.
   *
   * The NetCDF variable is declared along the following dimensions, from
   * innermost to outermost:
   *
   * @li the @c np dimension,
   *
   * @li if the model variable has dimensions, the NetCDF dimensions that
   * correspond to these,
   *
   * @li if the variable is dynamic (e.g. a state variable), the
   * @c nr dimension,
   *
   * @li if it exists, the @c ns dimension.
   */
  NcVar* createVar(const Var* var, const bool nr = false, const bool np = true);

  /**
   * Create variable in NetCDF file using the flexible format.
   *
   * @param var Dynamic variable in model for which to create variable in
   * NetCDF file.
   *
   * @return The variable.
   *
   * @li the @c npr unlimited dimension,
   *
   * @li if the model variable has dimensions, the NetCDF dimensions that
   * correspond to these,
   *
   * @li if it exists, the @c ns dimension.
   */
  NcVar* createFlexiVar(const Var* node);

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
   *
   * @return The variable.
   */
  NcVar* mapVar(const Var* node);

  /**
   * NetCDF file.
   */
  NcFile* ncFile;

  /**
   * File name. Used for reopening file with new file handle under copy
   * constructor.
   */
  std::string file;
};
}

#include "../misc/assert.hpp"

inline bool bi::NetCDFBuffer::hasDim(const char* name) {
  /* pre-condition */
  BI_ASSERT(name != NULL);

  NcDim* dim = ncFile->get_dim(name);

  return (dim != NULL && dim->is_valid());
}

inline bool bi::NetCDFBuffer::hasVar(const char* name) {
  /* pre-condition */
  BI_ASSERT(name != NULL);

  NcVar* var = ncFile->get_var(name);

  return (var != NULL && var->is_valid());
}

inline void bi::NetCDFBuffer::sync() {
  ncFile->sync();
}

#endif
