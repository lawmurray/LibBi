/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputBuffer.hpp"

using namespace bi;

SparseInputBufferState::SparseInputBufferState(const Model& m) :
    masks(NUM_VAR_TYPES) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks[type] = new mask_type(m.getNumVars(type));
  }
}

SparseInputBufferState::SparseInputBufferState(
    const SparseInputBufferState& o) : starts(o.starts.size()),
    lens(o.lens.size()), masks(NUM_VAR_TYPES) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks[type] = new mask_type(o.masks[type]->getNumVars());
  }
  operator=(o);
}

SparseInputBufferState& SparseInputBufferState::operator=(
    const SparseInputBufferState& o) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    *masks[type] = *o.masks[type];
  }
  starts = o.starts;
  lens = o.lens;

  return *this;
}

SparseInputBufferState::~SparseInputBufferState() {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    delete masks[type];
  }
}

SparseInputBuffer::SparseInputBuffer(const Model& m) : m(m),
    vDims(NUM_VAR_TYPES), masks0(NUM_VAR_TYPES), state(m) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    masks0[type] = new mask_type(m.getNumVars(type));
  }
}

void SparseInputBuffer::mark() {
  Markable<SparseInputBufferState>::mark(state);
}

void SparseInputBuffer::restore() {
  Markable<SparseInputBufferState>::restore(state);
}

void SparseInputBuffer::top() {
  Markable<SparseInputBufferState>::top(state);
}

void SparseInputBuffer::pop() {
  Markable<SparseInputBufferState>::pop();
}
