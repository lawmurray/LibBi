/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * This file provides a fairly thin wrapper around the NetCDF C Interface
 * designed to:
 *
 * @li provide error handling consistent with LibBi error reporting,
 * @li provide return values where convenient once error codes are handled
 * internally, and
 * @li provide generic or overloaded functions where convenient.
 *
 * Note that the older NetCDF C++ Interface does not support certain features
 * of NetCDF 4 that have become necessary in LibBi, while the newer interface
 * represents a significant change that has made it easier to refactor
 * LibBi's NetCDF code to simply use the C interface directly.
 */
#ifndef BI_NETCDF_NETCDF_HPP
#define BI_NETCDF_NETCDF_HPP

#include <netcdf.h>
#include <string>
#include <vector>

#ifdef ENABLE_SINGLE
#define NC_REAL NC_FLOAT
#else
#define NC_REAL NC_DOUBLE
#endif

namespace bi {
/**
 * @name Files
 */
//@{
/**
 * @ingroup io_netcdf
 */
int nc_open(const std::string& path, int mode);

/**
 * @ingroup io_netcdf
 */
int nc_create(const std::string& path, int cmode);

/**
 * @ingroup io_netcdf
 */
void nc_set_fill(int ncid, int fillmode);

/**
 * @ingroup io_netcdf
 */
void nc_sync(int ncid);

/**
 * @ingroup io_netcdf
 */
void nc_redef(int ncid);

/**
 * @ingroup io_netcdf
 */
void nc_enddef(int ncid);

/**
 * @ingroup io_netcdf
 */
void nc_close(int ncid);

/**
 * @ingroup io_netcdf
 */
int nc_inq_nvars(int ncid);
//@}

/**
 * @name Dimensions
 */
//@{
/**
 * @ingroup io_netcdf
 */
int nc_def_dim(int ncid, const std::string& name, size_t len);

/**
 * Define unlimited dimension.
 *
 * @ingroup io_netcdf
 *
 * @param ncid
 * @param name
 *
 * @return Dimension id.
 */
int nc_def_dim(int ncid, const std::string& name);

/**
 * @ingroup io_netcdf
 */
int nc_inq_dimid(int ncid, const std::string& name);

/**
 * @ingroup io_netcdf
 */
std::string nc_inq_dimname(int ncid, int dimid);

/**
 * @ingroup io_netcdf
 */
size_t nc_inq_dimlen(int ncid, int dimid);
//@}

/**
 * @name Variables
 */
//@{
/**
 * @ingroup io_netcdf
 */
int nc_def_var(int ncid, const std::string& name, nc_type xtype,
    const std::vector<int>& dimids);

/**
 * Define scalar variable.
 *
 * @ingroup io_netcdf
 *
 * @param ncid
 * @param name
 * @param xtype
 *
 * @return Variable id.
 */
int nc_def_var(int ncid, const std::string& name, nc_type xtype);

/**
 * Define vector variable.
 *
 * @ingroup io_netcdf
 *
 * @param ncid
 * @param name
 * @param xtype
 * @param dimid
 *
 * @return Variable id.
 */
int nc_def_var(int ncid, const std::string& name, nc_type xtype, int dimid);

/**
 * Define matrix variable.
 *
 * @ingroup io_netcdf
 *
 * @param ncid
 * @param name
 * @param xtype
 * @param dimid1
 * @param dimid2
 *
 * @return Variable id.
 */
int nc_def_var(int ncid, const std::string& name, nc_type xtype, int dimid1,
    int dimid2);

/**
 * @ingroup io_netcdf
 */
int nc_inq_varid(int ncid, const std::string& name);

/**
 * @ingroup io_netcdf
 */
std::string nc_inq_varname(int ncid, int varid);

/**
 * @ingroup io_netcdf
 */
int nc_inq_varndims(int ncid, int varid);

/**
 * @ingroup io_netcdf
 */
std::vector<int> nc_inq_vardimid(int ncid, int varid);
//@}

/**
 * @name Attributes
 */
//@{
/**
 * Put global attribute.
 *
 * @ingroup io_netcdf
 */
void nc_put_att(int ncid, const std::string& name, const std::string& value);

/**
 * @ingroup io_netcdf
 */
void nc_put_att(int ncid, const std::string& name, const int value);

/**
 * @ingroup io_netcdf
 */
void nc_put_att(int ncid, const std::string& name, const float value);

/**
 * @ingroup io_netcdf
 */
void nc_put_att(int ncid, const std::string& name, const double value);

//@}

/**
 * @name Reading and writing
 */
//@{
/**
 * @ingroup io_netcdf
 */
void nc_get_var(int ncid, int varid, int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var(int ncid, int varid, long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var(int ncid, int varid, float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var(int ncid, int varid, double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var(int ncid, int varid, const int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var(int ncid, int varid, const long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var(int ncid, int varid, const float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var(int ncid, int varid, const double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const size_t index, int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const size_t index, long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const size_t index, float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const size_t index, double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const size_t index, const int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const size_t index, const long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const size_t index, const float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const size_t index, const double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_var1(int ncid, int varid, const std::vector<size_t>& index,
    double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_var1(int ncid, int varid, const std::vector<size_t>& index,
    const double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const size_t start, const size_t count,
    int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const size_t start, const size_t count,
    long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const size_t start, const size_t count,
    float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const size_t start, const size_t count,
    double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const size_t start, const size_t count,
    const int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const size_t start, const size_t count,
    const long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const size_t start, const size_t count,
    const float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const size_t start, const size_t count,
    const double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_get_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, double* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const int* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const long* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const float* ip);

/**
 * @ingroup io_netcdf
 */
void nc_put_vara(int ncid, int varid, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const double* ip);
//@}

}

#endif
