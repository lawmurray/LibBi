/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputBuffer.hpp"

bi::SparseInputBufferState::SparseInputBufferState(const Model& m) :
    masks(NUM_VAR_TYPES) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks[type] = new mask_type(m.getNumVars(type));
  }
}

bi::SparseInputBufferState::SparseInputBufferState(
    const SparseInputBufferState& o) : starts(o.starts.size()),
    lens(o.lens.size()), masks(NUM_VAR_TYPES), times(o.times) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks[type] = new mask_type(o.masks[type]->getNumVars());
  }
  operator=(o);
}

bi::SparseInputBufferState& bi::SparseInputBufferState::operator=(
    const SparseInputBufferState& o) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    *masks[type] = *o.masks[type];
  }
  starts = o.starts;
  lens = o.lens;
  times = o.times;

  return *this;
}

bi::SparseInputBufferState::~SparseInputBufferState() {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    delete masks[type];
  }
}

bi::SparseInputBuffer::SparseInputBuffer(const Model& m) : m(m),
    vDims(NUM_VAR_TYPES), masks0(NUM_VAR_TYPES), state(m) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks0[type] = new mask_type(m.getNumVars(type));
  }
}

void bi::SparseInputBuffer::mark() {
  Markable<SparseInputBufferState>::mark(state);
}

void bi::SparseInputBuffer::restore() {
  Markable<SparseInputBufferState>::restore(state);
}

void bi::SparseInputBuffer::top() {
  Markable<SparseInputBufferState>::top(state);
}

void bi::SparseInputBuffer::pop() {
  Markable<SparseInputBufferState>::pop();
}
