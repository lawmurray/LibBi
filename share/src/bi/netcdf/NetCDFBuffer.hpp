/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_NETCDFBUFFER_HPP
#define BI_NETCDF_NETCDFBUFFER_HPP

#include "netcdf.hpp"
#include "../buffer/buffer.hpp"

namespace bi {
/**
 * NetCDF input or output file.
 *
 * @ingroup io_netcdf
 */
class NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  NetCDFBuffer(const std::string& file = "", const FileMode mode = READ_ONLY);

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
