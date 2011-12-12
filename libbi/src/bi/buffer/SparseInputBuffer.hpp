/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEINPUTBUFFER_HPP
#define BI_BUFFER_SPARSEINPUTBUFFER_HPP

#include "SparseMask.hpp"
#include "../model/BayesNet.hpp"
#include "../misc/Markable.hpp"
#include "../math/scalar.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

#include <map>

namespace bi {
/**
 * @internal
 *
 * State of SparseInputBuffer.
 *
 * @ingroup io
 */
struct SparseInputBufferState {
  /**
   * Integral vector type for ids in dense and sparse masks.
   */
  typedef host_vector_temp_type<int>::type vector_type;

  /**
   * Mask type.
   */
  typedef SparseMask<> mask_type;

  /**
   * Constructor.
   */
  SparseInputBufferState();

  /**
   * Copy constructor.
   */
  SparseInputBufferState(const SparseInputBufferState& o);

  /**
   * Current offset into each record dimension.
   */
  vector_type starts;

  /**
   * Current length over each record dimension.
   */
  vector_type lens;

  /**
   * Mask of active nodes at the current time, indexed by type.
   */
  std::vector<mask_type> masks;

  /**
   * Time variable ids keyed by their current time.
   */
  std::multimap<real,int> times;
};
}

namespace bi {
/**
 * Buffer for storing and sequentially reading input in sparse format.
 *
 * @ingroup io
 */
class SparseInputBuffer : public Markable<SparseInputBufferState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param flags Flags to pass to InputBuffer constructor.
   */
  SparseInputBuffer(const BayesNet& m);

  /**
   * @copydoc concept::InputBuffer::getTime()
   */
  real getTime() const;

  /**
   * @copydoc concept::InputBuffer::size()
   *
   * The total number of active variables at the current time.
   */
  int size(const NodeType type) const;

  /**
   * @copydoc concept::InputBuffer::size0()
   *
   * The total number of active static variables.
   */
  int size0(const NodeType type) const;

  /**
   * @copydoc concept::InputBuffer::getDenseMask()
   */
  const SparseInputBufferState::mask_type& getMask(const NodeType type) const;

  /**
   * @copydoc concept::InputBuffer::getMask0()
   */
  const SparseInputBufferState::mask_type& getMask0(const NodeType type) const;

  /**
   * @copydoc concept::InputBuffer::isValid()
   */
  bool isValid() const;

  /**
   * Is time variable associated with at least one variable?
   */
  bool isAssoc(const int tVar) const;

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

protected:
  /**
   * Model.
   */
  const BayesNet& m;

  /**
   * Record dimension to time variable associations, indexed by record
   * dimension id. Value of -1 for record dimensions not associated with
   * a time variable.
   */
  std::vector<int> tAssoc;

  /**
   * Record dimension to coordinate variable associations, indexed by record
   * dimension id. Value of -1 for record dimensions not associated with
   * a coordinate variable.
   */
  std::vector<int> cAssoc;

  /**
   * Record dimension to model variable associations, indexed by record
   * dimension id and type.
   */
  std::vector<std::vector<std::list<int> > > vAssoc;

  /**
   * Model variables not associated with record dimension, indexed by type.
   */
  std::vector<std::list<int> > vUnassoc;

  /**
   * Time variable to record dimension associations, indexed by time variable
   * id.
   */
  std::vector<int> tDims;

  /**
   * Model variable to record dimension associations, indexed by type and
   * variable id. Value of -1 for model variables not associated with a
   * record dimension.
   */
  std::vector<std::vector<int> > vDims;

  /**
   * Mask of active nodes that are not associated with a time variable,
   * indexed by type.
   */
  std::vector<SparseInputBufferState::mask_type> masks0;

  /**
   * Current state of buffer.
   */
  SparseInputBufferState state;
};
}

inline real bi::SparseInputBuffer::getTime() const {
  return state.times.begin()->first;
}

inline int bi::SparseInputBuffer::size(const NodeType type) const {
  return state.masks[type].size();
}

inline int bi::SparseInputBuffer::size0(const NodeType type) const {
  return masks0[type].size();
}

inline const bi::SparseInputBufferState::mask_type&
    bi::SparseInputBuffer::getMask(const NodeType type) const {
  return state.masks[type];
}

inline const bi::SparseInputBufferState::mask_type&
    bi::SparseInputBuffer::getMask0(const NodeType type) const {
  return masks0[type];
}

inline bool bi::SparseInputBuffer::isValid() const {
  return !state.times.empty();
}

inline bool bi::SparseInputBuffer::isAssoc(const int tVar) const {
  int rDim = tDims[tVar];
  unsigned i;
  bool result = false;
  for (i = 0; !result && i < vAssoc[rDim].size(); ++i) {
    result = vAssoc[rDim][i].size() > 0;
  }
  return result;
}

#endif
