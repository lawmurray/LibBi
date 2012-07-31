/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Coord.hpp"

using namespace bi;

Coord::Coord(const int i, const Var* var) : coords(var->getNumDims()),
    sizes(var->getNumDims()) {
  int j = i, k;
  int rest;

  for (k = coords.size() - 1; k >= 0; --k) {
    sizes[k] = var->getDim(k)->getSize();
    rest = j/sizes[k];
    coords[k] = j - rest;
    j = rest;
  }
}

void Coord::inc() {
  int k;

  for (k = coords.size() - 1; k >= 0; --k) {
    if (coords[k] < sizes[k] - 1) {
      ++coords[k];
      break;
    } else {
      coords[k] = 0;
    }
  }
}

void Coord::dec() {
  int k;

  for (k = coords.size() - 1; k >= 0; --k) {
    if (coords[k] > 0) {
      --coords[k];
      break;
    } else {
      coords[k] = sizes[k] - 1;
    }
  }
}

int Coord::index() const {
  int k, i, len;

  if (coords.size() > 0) {
    i = coords.back();
    len = sizes.back();

    for (k = coords.size() - 2; k >= 0; --k) {
      i += coords[k]*len;
      len *= sizes[k];
    }
  } else {
    i = 0;
  }
  return i;
}
