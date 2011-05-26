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
