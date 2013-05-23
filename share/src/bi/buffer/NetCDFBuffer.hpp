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
#include "../math/scalar.hpp"

#include "netcdfcpp.h"

/**
 * NetCDF type identifier for real.
 *
 * @ingroup io_buffer
 */
extern NcType netcdf_real;

namespace bi {
/**
 * Result buffer supported by NetCDF file.
 *
 * @ingroup io_buffer
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

  /**
   * Does nothing but maintain interface with caches.
   */
  void clear();

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
  ~NetCDFBuffer();

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
   * @param var Variable in model for which to create variable in NetCDF
   * file.
   * @param nr Declare the variable along the @c nr dimension?
   * @param np Declare the variable along the @c np dimension?
   *
   * @return The variable.
   *
   * The NetCDF variable is declared along the following dimensions, from
   * innermost to outermost:
   *
   * @li the @c np dimension, if desired,
   *
   * @li if the model variable has dimensions, the NetCDF dimensions that
   * correspond to these,
   *
   * @li the @c nr dimension, if desired,
   *
   * @li the @c ns dimension, if it exists.
   */
  NcVar* createVar(const Var* var, const bool nr, const bool np);

  /**
   * Create variable in NetCDF file using the flexible format.
   *
   * @param var Variable in model for which to create variable in NetCDF
   * file.
   *
   * @return The variable.
   *
   * The NetCDF variable is declared along the following dimensions, from
   * innermost to outermost:
   *
   * @li the @c npr unlimited dimension,
   *
   * @li if the model variable has dimensions, the NetCDF dimensions that
   * correspond to these,
   *
   * @li the @c ns dimension, if it exists.
   */
  NcVar* createFlexiVar(const Var* var);

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
   * @param var Variable in model for which to map variable in NetCDF file.
   *
   * @return The variable.
   */
  NcVar* mapVar(const Var* var);

  /**
   * Read from one-dimensional variable.
   *
   * @tparam V1 Vector type.
   *
   * @param ncVar NetCDF variable.
   * @param offset Offset into variable.
   * @param[out] x Output.
   */
  template<class V1>
  void read1d(NcVar* ncVar, const int offset, V1 x) const;

  /**
   * Write into one-dimensional variable.
   *
   * @tparam V1 Vector type.
   *
   * @param ncVar NetCDF variable.
   * @param offset Offset into variable.
   * @param x Input.
   */
  template<class V1>
  void write1d(NcVar* ncVar, const int offset, const V1 x);

  /**
   * Read scalar from variable.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param[out] x Scalar.
   */
  void readScalar(NcVar* ncVar, const int k, real& x) const;

  /**
   * Write scalar to variable.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param x Scalar.
   */
  void writeScalar(NcVar* ncVar, const int k, const real& x);

  /**
   * Read vector from variable.
   *
   * @tparam V1 Vector type.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param[out] x Vector.
   */
  template<class V1>
  void readVector(NcVar* ncVar, const int k, V1 x) const;

  /**
   * Write vector to variable.
   *
   * @tparam V1 Vector type.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param x Vector.
   */
  template<class V1>
  void writeVector(NcVar* ncVar, const int k, const V1 x);

  /**
   * Read matrix to variable.
   *
   * @tparam M1 Matrix type.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param[out] X Matrix.
   */
  template<class M1>
  void readMatrix(NcVar* ncVar, const int k, M1 X) const;

  /**
   * Write matrix to variable.
   *
   * @tparam M1 Matrix type.
   *
   * @param ncVar NetCDF variable.
   * @param k Index along outer dimension.
   * @param X Matrix.
   */
  template<class M1>
  void writeMatrix(NcVar* ncVar, const int k, const M1 X);

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
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

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

template<class V1>
void bi::NetCDFBuffer::read1d(NcVar* ncVar, const int offset, V1 x) const {
  /* pre-condition */
  BI_ASSERT(ncVar->num_dims() == 1);
  BI_ASSERT(offset >= 0 && offset + x.size() <= ncVar->get_dim(0)->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = ncVar->set_cur(offset);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (V1::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    ret = ncVar->get(x1.buf(), x1.size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    x = x1;
  } else {
    ret = ncVar->get(x.buf(), x.size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V1>
void bi::NetCDFBuffer::write1d(NcVar* ncVar, const int offset, const V1 x) {
  /* pre-condition */
  BI_ASSERT(ncVar->num_dims() == 1);
  BI_ASSERT(offset >= 0 && offset + x.size() <= ncVar->get_dim(0)->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = ncVar->set_cur(offset);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());

  if (V1::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    x1 = x;
    synchronize(V1::on_device);
    ret = ncVar->put(x1.buf(), x1.size());
  } else {
    ret = ncVar->put(x.buf(), x.size());
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
}

template<class V1>
void bi::NetCDFBuffer::readVector(NcVar* ncVar, const int k, V1 x) const {
  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  long offsets[] = { k, 0 };
  long counts[] = { 1, x.size() };
  BI_UNUSED NcBool ret;

  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (V1::on_device || !x.contiguous()) {
    temp_vector_type tmp(x.size());
    ret = ncVar->get(tmp.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    x = tmp;
  } else {
    ret = ncVar->get(x.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V1>
void bi::NetCDFBuffer::writeVector(NcVar* ncVar, const int k, const V1 x) {
  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  long offsets[] = { k, 0 };
  long counts[] = { 1, x.size() };
  BI_UNUSED NcBool ret;

  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());
  if (V1::on_device || !x.contiguous()) {
    temp_vector_type tmp(x.size());
    tmp = x;
    synchronize(V1::on_device);
    ret = ncVar->put(tmp.buf(), counts);
  } else {
    ret = ncVar->put(x.buf(), counts);
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
}

template<class M1>
void bi::NetCDFBuffer::readMatrix(NcVar* ncVar, const int k, M1 X) const {
  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, X.size2(), X.size1() };
  BI_UNUSED NcBool ret;

  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
  if (M1::on_device || !X.contiguous()) {
    temp_matrix_type tmp(X.size1(), X.size2());
    ret = ncVar->get(tmp.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    X = tmp;
  } else {
    ret = ncVar->get(X.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class M1>
void bi::NetCDFBuffer::writeMatrix(NcVar* ncVar, const int k, const M1 X) {
  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, X.size2(), X.size1() };
  BI_UNUSED NcBool ret;

  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());
  if (M1::on_device || !X.contiguous()) {
    temp_matrix_type tmp(X.size1(), X.size2());
    tmp = X;
    synchronize(M1::on_device);
    ret = ncVar->put(tmp.buf(), counts);
  } else {
    ret = ncVar->put(X.buf(), counts);
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
}

#endif
