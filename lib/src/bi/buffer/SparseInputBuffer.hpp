/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEINPUTBUFFER_HPP
#define BI_BUFFER_SPARSEINPUTBUFFER_HPP

#include "../model/BayesNet.hpp"
#include "../misc/Markable.hpp"
#include "../math/scalar.hpp"

#include <map>

namespace bi {
/**
 * State of SparseInputBuffer.
 */
struct SparseInputBufferState {
  /**
   * Constructor.
   */
  SparseInputBufferState();

  /**
   * Current time.
   */
  real t;

  /**
   * Ids of nodes that have changed at the current time, indexed by type.
   */
  std::vector<std::vector<int> > current;

  /**
   * Time variable indices keyed by their next time.
   */
  std::multimap<real,int> nextTimes;

  /**
   * Current offset into time dimension for each time variable.
   */
  std::vector<int> nrs;
};
}

inline bi::SparseInputBufferState::SparseInputBufferState() : t(0.0),
    current(NUM_NODE_TYPES) {
  //
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
  real getTime();

  /**
   * @copydoc concept::InputBuffer::hasNext()
   */
  bool hasNext();

  /**
   * @copydoc concept::InputBuffer::getNextTime()
   */
  real getNextTime();

  /**
   * @copydoc concept::InputBuffer::countCurrentNodes()
   */
  int countCurrentNodes(const NodeType type);

  /**
   * @copydoc concept::InputBuffer::countNextNodes()
   */
  int countNextNodes(const NodeType type);

  /**
   * @copydoc concept::InputBuffer::getCurrentNodes()
   */
  template<class V1>
  void getCurrentNodes(const NodeType type, V1& ids);

  /**
   * @copydoc concept::InputBuffer::getNextNodes()
   */
  template<class V1>
  void getNextNodes(const NodeType type, V1& ids);

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
   * Next time variable.
   *
   * @return Index of the time variable with the next time in the file.
   */
  int getNextTimeVar();

  /**
   * Erase next time.
   */
  void eraseNextTime();

  /**
   * Model.
   */
  const BayesNet& m;

  /**
   * Time dimension to variable associations, indexed by time dimension id
   * and type.
   */
  std::vector<std::vector<std::vector<int> > > assoc;

  /**
   * Reverse associations: variable to time dimension associations, indexed
   * by type and other variable id. Value of -1 for variables not associated
   * with a time dimension.
   */
  std::vector<std::vector<int> > reverseAssoc;

  /**
   * Variables not associated with a time dimension, indexed by type.
   */
  std::vector<std::vector<int> > unassoc;

  /**
   * Current state of buffer.
   */
  SparseInputBufferState state;
};
}

#include <algorithm>

inline void bi::SparseInputBuffer::eraseNextTime() {
  state.nextTimes.erase(state.nextTimes.begin());
}

inline bool bi::SparseInputBuffer::hasNext() {
  return !state.nextTimes.empty();
}

inline real bi::SparseInputBuffer::getTime() {
  return state.t;
}

inline real bi::SparseInputBuffer::getNextTime() {
  return state.nextTimes.begin()->first;
}

inline int bi::SparseInputBuffer::getNextTimeVar() {
  return state.nextTimes.begin()->second;
}

inline int bi::SparseInputBuffer::countCurrentNodes(const NodeType type) {
  return state.current[type].size();
}

inline int bi::SparseInputBuffer::countNextNodes(const NodeType type) {
  typedef std::multimap<real,int>::iterator tv_iterator;

  std::pair<tv_iterator,tv_iterator> tvRange;
  int count = 0;

  tvRange = state.nextTimes.equal_range(state.nextTimes.begin()->first);
  while (tvRange.first != tvRange.second) {
    /* count nodes for this time variable */
    std::vector<int>& vec = assoc[tvRange.first->second][type];
    count += vec.size();
    ++tvRange.first;
  }

  return count;
}

template<class V1>
void bi::SparseInputBuffer::getCurrentNodes(const NodeType type,
    V1& ids) {
  ids.resize(state.current[type].size());
  std::copy(state.current[type].begin(), state.current[type].end(), ids.begin());
}

template<class V1>
void bi::SparseInputBuffer::getNextNodes(const NodeType type, V1& ids) {
  typedef std::multimap<real,int>::iterator tv_iterator;

  std::pair<tv_iterator,tv_iterator> tvRange;
  ids.clear();

  /* next time vars */
  tvRange = state.nextTimes.equal_range(state.nextTimes.begin()->first);
  while (tvRange.first != tvRange.second) {
    /* nodes for this time variable */
    std::vector<int>& vec = assoc[tvRange.first->second][type];
    ids.resize(ids.size() + vec.size());
    std::copy(vec.begin(), vec.end(), ids.end() - vec.size());
    ++tvRange.first;
  }

  /* sort */
  std::sort(ids.begin(), ids.end());
}

#endif
