/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputBuffer.hpp"

using namespace bi;

SparseInputBufferState::SparseInputBufferState() : masks(NUM_VAR_TYPES) {
  //
}

SparseInputBufferState::SparseInputBufferState(
    const SparseInputBufferState& o) : starts(o.starts.size()),
    lens(o.lens.size()), masks(o.masks), times(o.times) {
  starts = o.starts;
  lens = o.lens;
}

SparseInputBuffer::SparseInputBuffer(const Model& m) : m(m),
    vDims(NUM_VAR_TYPES), masks0(NUM_VAR_TYPES) {
  VarType type;
  int i;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    state.masks[type].resize(m.getNumVars(type));
    masks0[type].resize(m.getNumVars(type));
  }
}

void SparseInputBuffer::mark() {
  Markable<SparseInputBufferState>::mark(state);
}

void SparseInputBuffer::restore() {
  Markable<SparseInputBufferState>::restore(state);
}
