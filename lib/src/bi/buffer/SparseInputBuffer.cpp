/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputBuffer.hpp"

using namespace bi;

SparseInputBuffer::SparseInputBuffer(const BayesNet& m) : m(m),
    unassoc(NUM_NODE_TYPES) {
  //
}

void SparseInputBuffer::mark() {
  Markable<SparseInputBufferState>::mark(state);
}

void SparseInputBuffer::restore() {
  Markable<SparseInputBufferState>::restore(state);
}
