/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputBuffer.hpp"

using namespace bi;

SparseInputBufferState::SparseInputBufferState() : masks(NUM_NODE_TYPES) {
  //
}

SparseInputBufferState::SparseInputBufferState(
    const SparseInputBufferState& o) : starts(o.starts.size()),
    lens(o.lens.size()), masks(o.masks), times(o.times) {
  starts = o.starts;
  lens = o.lens;
}

SparseInputBuffer::SparseInputBuffer(const BayesNet& m) : m(m),
    vDims(NUM_NODE_TYPES), masks0(NUM_NODE_TYPES) {
  //
}

void SparseInputBuffer::mark() {
  Markable<SparseInputBufferState>::mark(state);
}

void SparseInputBuffer::restore() {
  Markable<SparseInputBufferState>::restore(state);
}
