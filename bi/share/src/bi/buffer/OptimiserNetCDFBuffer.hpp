/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_OPTIMISERNETCDFBUFFER_HPP
#define BI_BUFFER_OPTIMISERNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of
 * NelderMeadOptimiser.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::OptimiserBuffer
 */
class OptimiserNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  OptimiserNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Destructor.
   */
  virtual ~OptimiserNetCDFBuffer();

  /**
   * @copydoc concept::OptimiserBuffer::size()
   */
  int size() const;

  /**
   * @copydoc concept::OptimiserBuffer::readState()
   */
  template<class V1>
  void readState(const VarType type, const int k, V1 x);

  /**
   * @copydoc concept::OptimiserBuffer::writeState()
   */
  template<class V1>
  void writeState(const VarType type, const int k, const V1 x);

  /**
   * @copydoc concept::OptimiserBuffer::readValue()
   */
  void readValue(const int k, real& x);

  /**
   * @copydoc concept::OptimiserBuffer::writeValue()
   */
  void writeValue(const int k, const real& x);

  /**
   * @copydoc concept::OptimiserBuffer::readSize()
   */
  void readSize(const int k, real& x);

  /**
   * @copydoc concept::OptimiserBuffer::writeSize()
   */
  void writeSize(const int k, const real& x);

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
   * Model.
   */
  const Model& m;

  /**
   * Record dimension.
   */
  NcDim* nsDim;

  /**
   * Model dimensions.
   */
  std::vector<NcDim*> nDims;

  /**
   * Node variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;

  /**
   * Function value variable.
   */
  NcVar* valueVar;

  /**
   * Size variable.
   */
  NcVar* sizeVar;
};
}

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"

inline int bi::OptimiserNetCDFBuffer::size() const {
  return nsDim->size();
}

template<class V1>
void bi::OptimiserNetCDFBuffer::readState(const VarType type, const int k,
    V1 x) {
  /* pre-condition */
  assert (k >= 0 && k < nsDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->getIO()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      assert (ncVar->get_dim(j) == nsDim);
      offsets[j] = k;
      counts[j] = 1;
      ++j;

      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT(ret, "Indexing out of bounds reading " << ncVar->name());

      if (V1::on_device || x.inc() != 1) {
        temp_vector_type x1(size);
        ret = ncVar->get(x1.buf(), counts.buf());
        BI_ASSERT(ret, "Inconvertible type reading " << ncVar->name());
        subrange(x, start, size) = x1;
      } else {
        ret = ncVar->get(subrange(x, start, size).buf(), counts.buf());
        BI_ASSERT(ret, "Inconvertible type reading " << ncVar->name());
      }
    }
  }
}

template<class V1>
void bi::OptimiserNetCDFBuffer::writeState(const VarType type, const int k,
    const V1 x) {
  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->getIO()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      assert (ncVar->get_dim(j) == nsDim);
      offsets[j] = k;
      counts[j] = 1;
      ++j;

      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT(ret, "Indexing out of bounds writing " << ncVar->name());

      if (V1::on_device || x.inc() != 1) {
        temp_vector_type x1(size);
        x1 = subrange(x, start, size);
        synchronize(V1::on_device);
        ret = ncVar->put(x1.buf(), counts.buf());
      } else {
        ret = ncVar->put(subrange(x, start, size).buf(), counts.buf());
      }
      BI_ASSERT(ret, "Inconvertible type reading " << ncVar->name());
    }
  }
}

#endif
