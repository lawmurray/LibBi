/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "netcdf.hpp"

#include "../misc/assert.hpp"
#include "../misc/compile.hpp"

int bi::nc_open(const std::string& path, int mode) {
  int ncid, status;
  status = ::nc_open(path.c_str(), mode, &ncid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not open " << path);
  return ncid;
}

int bi::nc_create(const std::string& path, int cmode) {
  int ncid, status;
  status = ::nc_create(path.c_str(), cmode, &ncid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not create " << path);
  return ncid;
}

void bi::nc_set_fill(int ncid, int fillmode) {
  int status = ::nc_set_fill(ncid, fillmode, NULL);
  BI_WARN_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_sync(int ncid) {
  int status = ::nc_sync(ncid);
  BI_WARN_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_redef(int ncid) {
  int status = ::nc_redef(ncid);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_enddef(int ncid) {
  int status = ::nc_enddef(ncid);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_close(int ncid) {
  int status = ::nc_close(ncid);
  BI_WARN_MSG(status == NC_NOERR, nc_strerror(status));
}

int bi::nc_inq_nvars(int ncid) {
  int nvars, status;
  status = ::nc_inq_nvars(ncid, &nvars);
  BI_ERROR_MSG(status == NC_NOERR, "Could not determine number of variables");
  return nvars;
}

int bi::nc_def_dim(int ncid, const std::string& name, size_t len) {
  int dimid, status;
  status = ::nc_def_dim(ncid, name.c_str(), len, &dimid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define dimension " << name);
  return dimid;
}

int bi::nc_def_dim(int ncid, const std::string& name) {
  int dimid, status;
  status = ::nc_def_dim(ncid, name.c_str(), NC_UNLIMITED, &dimid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define dimension " << name);
  return dimid;
}

int bi::nc_inq_dimid(int ncid, const std::string& name) {
  int dimid = -1;
  BI_UNUSED int status;
  status = ::nc_inq_dimid(ncid, name.c_str(), &dimid);
  return dimid;
}

std::string bi::nc_inq_dimname(int ncid, int dimid) {
  char name[NC_MAX_NAME + 1];
  int status;
  status = ::nc_inq_dimname(ncid, dimid, name);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
  return std::string(name);
}

size_t bi::nc_inq_dimlen(int ncid, int dimid) {
  size_t len;
  int status;
  status = ::nc_inq_dimlen(ncid, dimid, &len);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
  return len;
}

int bi::nc_def_var(int ncid, const std::string& name, nc_type xtype,
    const std::vector<int>& dimids) {
  int varid, status;
  status = ::nc_def_var(ncid, name.c_str(), xtype, dimids.size(),
      dimids.data(), &varid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define variable " << name);
  return varid;
}

int bi::nc_def_var(int ncid, const std::string& name, nc_type xtype) {
  int varid, status;
  status = ::nc_def_var(ncid, name.c_str(), xtype, 0, NULL, &varid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define variable " << name);
  return varid;
}

int bi::nc_def_var(int ncid, const std::string& name, nc_type xtype,
    int dimid) {
  int varid, status;
  status = ::nc_def_var(ncid, name.c_str(), xtype, 1, &dimid, &varid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define variable " << name);
  return varid;
}

int bi::nc_def_var(int ncid, const std::string& name, nc_type xtype,
    int dimid1, int dimid2) {
  int varid, status;
  int dims[2] = { dimid1, dimid2 };
  status = ::nc_def_var(ncid, name.c_str(), xtype, 2, dims, &varid);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define variable " << name);
  return varid;
}

int bi::nc_inq_varid(int ncid, const std::string& name) {
  int varid = -1;
  BI_UNUSED int status;
  status = ::nc_inq_varid(ncid, name.c_str(), &varid);
  return varid;
}

std::string bi::nc_inq_varname(int ncid, int varid) {
  char name[NC_MAX_NAME + 1];
  int status;
  status = ::nc_inq_varname(ncid, varid, name);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
  return std::string(name);
}

int bi::nc_inq_varndims(int ncid, int varid) {
  int ndims, status;
  status = ::nc_inq_varndims(ncid, varid, &ndims);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
  return ndims;
}

std::vector<int> bi::nc_inq_vardimid(int ncid, int varid) {
  int ndims = nc_inq_varndims(ncid, varid);
  std::vector<int> dimids(ndims);
  if (ndims > 0) {
    int status = ::nc_inq_vardimid(ncid, varid, dimids.data());
    BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
  }
  return dimids;
}

void bi::nc_put_att(int ncid, const std::string& name,
    const std::string& value) {
  int status = ::nc_put_att_text(ncid, NC_GLOBAL, name.c_str(),
      value.length(), value.c_str());
  BI_ERROR_MSG(status == NC_NOERR, "Could not define attribute " << name);
}

void bi::nc_put_att(int ncid, const std::string& name, const int value) {
  int status = ::nc_put_att_int(ncid, NC_GLOBAL, name.c_str(), NC_INT, 1,
      &value);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define attribute " << name);
}

void bi::nc_put_att(int ncid, const std::string& name, const float value) {
  int status = ::nc_put_att_float(ncid, NC_GLOBAL, name.c_str(), NC_FLOAT, 1,
      &value);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define attribute " << name);
}

void bi::nc_put_att(int ncid, const std::string& name, const double value) {
  int status = ::nc_put_att_double(ncid, NC_GLOBAL, name.c_str(), NC_DOUBLE,
      1, &value);
  BI_ERROR_MSG(status == NC_NOERR, "Could not define attribute " << name);
}

void bi::nc_get_var(int ncid, int varid, int* ip) {
  int status = ::nc_get_var_int(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var(int ncid, int varid, long* ip) {
  int status = ::nc_get_var_long(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var(int ncid, int varid, float* ip) {
  int status = ::nc_get_var_float(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var(int ncid, int varid, double* ip) {
  int status = ::nc_get_var_double(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var(int ncid, int varid, const int* ip) {
  int status = ::nc_put_var_int(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var(int ncid, int varid, const long* ip) {
  int status = ::nc_put_var_long(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var(int ncid, int varid, const float* ip) {
  int status = ::nc_put_var_float(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var(int ncid, int varid, const double* ip) {
  int status = ::nc_put_var_double(ncid, varid, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const size_t index, int* ip) {
  int status;
  status = ::nc_get_var1_int(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const size_t index, long* ip) {
  int status;
  status = ::nc_get_var1_long(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const size_t index, float* ip) {
  int status = ::nc_get_var1_float(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const size_t index, double* ip) {
  int status = ::nc_get_var1_double(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const size_t index,
    const int* ip) {
  int status = ::nc_put_var1_int(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const size_t index,
    const long* ip) {
  int status = ::nc_put_var1_long(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const size_t index,
    const float* ip) {
  int status = ::nc_put_var1_float(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const size_t index,
    const double* ip) {
  int status = ::nc_put_var1_double(ncid, varid, &index, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    int* ip) {
  int status;
  status = ::nc_get_var1_int(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    long* ip) {
  int status;
  status = ::nc_get_var1_long(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    float* ip) {
  int status = ::nc_get_var1_float(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    double* ip) {
  int status = ::nc_get_var1_double(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const int* ip) {
  int status = ::nc_put_var1_int(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const long* ip) {
  int status = ::nc_put_var1_long(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const float* ip) {
  int status = ::nc_put_var1_float(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const double* ip) {
  int status = ::nc_put_var1_double(ncid, varid, index.data(), ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const size_t start,
    const size_t count, int* ip) {
  int status = ::nc_get_vara_int(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const size_t start,
    const size_t count, long* ip) {
  int status = ::nc_get_vara_long(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const size_t start,
    const size_t count, float* ip) {
  int status = ::nc_get_vara_float(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const size_t start,
    const size_t count, double* ip) {
  int status = ::nc_get_vara_double(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const size_t start,
    const size_t count, const int* ip) {
  int status = ::nc_put_vara_int(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const size_t start,
    const size_t count, const long* ip) {
  int status = ::nc_put_vara_long(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const size_t start,
    const size_t count, const float* ip) {
  int status = ::nc_put_vara_float(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const size_t start,
    const size_t count, const double* ip) {
  int status = ::nc_put_vara_double(ncid, varid, &start, &count, ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, int* ip) {
  int status = ::nc_get_vara_int(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, long* ip) {
  int status = ::nc_get_vara_long(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, float* ip) {
  int status = ::nc_get_vara_float(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, double* ip) {
  int status = ::nc_get_vara_double(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const int* ip) {
  int status = ::nc_put_vara_int(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const long* ip) {
  int status = ::nc_put_vara_long(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const float* ip) {
  int status = ::nc_put_vara_float(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}

void bi::nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const double* ip) {
  int status = ::nc_put_vara_double(ncid, varid, start.data(), count.data(),
      ip);
  BI_ERROR_MSG(status == NC_NOERR, nc_strerror(status));
}
