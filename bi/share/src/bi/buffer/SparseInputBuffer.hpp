/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEINPUTBUFFER_HPP
#define BI_BUFFER_SPARSEINPUTBUFFER_HPP

#include "Mask.hpp"
#include "../model/Model.hpp"
#include "../misc/Markable.hpp"
#include "../math/scalar.hpp"

#include <map>

namespace bi {
/**
 * @internal
 *
 * State of SparseInputBuffer.
 *
 * @ingroup io_buffer
 */
struct SparseInputBufferState {
  /**
   * Integral vector type for ids in dense and sparse masks.
   */
  typedef temp_host_vector<int>::type vector_type;

  /**
   * Mask type.
   */
  typedef Mask<ON_HOST> mask_type;

  /**
   * Constructor.
   *
   * @param m Model.
   */
  SparseInputBufferState(const Model& m);

  /**
   * Copy constructor.
   */
  SparseInputBufferState(const SparseInputBufferState& o);

  /**
   * Destructor.
   */
  ~SparseInputBufferState();

  /**
   * Assignment operator.
   */
  SparseInputBufferState& operator=(const SparseInputBufferState& o);

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
  std::vector<mask_type*> masks;

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
 * @ingroup io_buffer
 */
class SparseInputBuffer : public Markable<SparseInputBufferState> {
public:
  /**
   * Mask type.
   */
  typedef SparseInputBufferState::mask_type mask_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param flags Flags to pass to InputBuffer constructor.
   */
  SparseInputBuffer(const Model& m);

  /**
   * Get the time for the next update.
   *
   * @return Time.
   */
  real getTime() const;

  /**
   * The total number of active variables of a given type at the current
   * time.
   *
   * @param type Variable type.
   *
   * @return Number of active variables.
   */
  int size(const VarType type) const;

  /**
   * The total number of active static variables of a given type.
   *
   * @param type Variable type.
   *
   * @return Number of active variables.
   */
  int size0(const VarType type) const;

  /**
   * Get the mask of active variables at the current time.
   *
   * @param type Variable type.
   *
   * @return Mask.
   */
  const SparseInputBufferState::mask_type& getMask(const VarType type) const;

  /**
   * Get the mask of active static variables.
   *
   * @param type Variable type.
   *
   * @return Mask.
   */
  const SparseInputBufferState::mask_type& getMask0(const VarType type) const;

  /**
   * Does any data remain?
   */
  bool isValid() const;

  /**
   * Is time variable associated with at least one variable?
   *
   * @param tVar Index of time variable.
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

  /**
   * @copydoc concept::Markable::top()
   */
  void top();

  /**
   * @copydoc concept::Markable::pop()
   */
  void pop();

protected:
  /**
   * Model.
   */
  const Model& m;

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
  std::vector<std::vector<std::vector<int> > > vAssoc;

  /**
   * Model variables not associated with record dimension, indexed by type.
   */
  std::vector<std::vector<int> > vUnassoc;

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
  std::vector<mask_type*> masks0;

  /**
   * Current state of buffer.
   */
  SparseInputBufferState state;
};
}

inline real bi::SparseInputBuffer::getTime() const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return state.times.begin()->first;
}

inline int bi::SparseInputBuffer::size(const VarType type) const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return state.masks[type]->size();
}

inline int bi::SparseInputBuffer::size0(const VarType type) const {
  return masks0[type]->size();
}

inline const bi::SparseInputBuffer::mask_type&
    bi::SparseInputBuffer::getMask(const VarType type) const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return *state.masks[type];
}

inline const bi::SparseInputBuffer::mask_type&
    bi::SparseInputBuffer::getMask0(const VarType type) const {
  return *masks0[type];
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
