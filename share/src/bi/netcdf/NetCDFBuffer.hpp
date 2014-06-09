/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_NETCDFBUFFER_HPP
#define BI_NETCDF_NETCDFBUFFER_HPP

#include "../model/Model.hpp"
#include "../method/misc.hpp"
#include "../math/scalar.hpp"

#include "netcdf.hpp"

namespace bi {
/**
 * NetCDF input or output file.
 *
 * @ingroup io_netcdf
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
  ~NetCDFBuffer();

  /**
   * Does nothing but maintain interface with caches.
   */
  void clear();

protected:
  /**
   * NetCDF file name recorded by constructor. Using this is preferred to the
   * nc_inq_path() function, as the latter requires fiddling with buffer
   * sizes to retrieve.
   */
  std::string file;

  /**
   * NetCDF file id.
   */
  int ncid;
};
}

#endif
