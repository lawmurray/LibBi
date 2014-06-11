/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "NetCDFBuffer.hpp"

#include "../misc/assert.hpp"

bi::NetCDFBuffer::NetCDFBuffer(const std::string& file, const FileMode mode) :
    file(file), ncid(-1) {
  BI_ERROR_MSG(!file.empty(), "No file specified");
  switch (mode) {
  case WRITE:
    ncid = nc_open(file, NC_WRITE);
    break;
  case NEW:
    ncid = nc_create(file, NC_NETCDF4 | NC_NOCLOBBER);
    nc_set_fill(ncid, NC_NOFILL);
    break;
  case REPLACE:
    ncid = nc_create(file, NC_NETCDF4);
    nc_set_fill(ncid, NC_NOFILL);
    break;
  default:
    ncid = nc_open(file, NC_NOWRITE);
  }
}

bi::NetCDFBuffer::NetCDFBuffer(const NetCDFBuffer& o) :
    file(o.file), ncid(-1) {
  if (!file.empty()) {
    ncid = nc_open(file, NC_NOWRITE);
  }
}

bi::NetCDFBuffer::~NetCDFBuffer() {
  nc_sync(ncid);
  nc_close(ncid);
}

void bi::NetCDFBuffer::clear() {
  //
}
