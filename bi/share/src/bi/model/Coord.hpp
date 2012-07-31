/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_COORD_HPP
#define BI_MODEL_COORD_HPP

#include "Var.hpp"

#include <vector>

namespace bi {
/**
 * Coordinate into variable.
 *
 * @ingroup model_high
 */
class Coord {
public:
  /**
   * Construct coordinate from serial index.
   *
   * @param i Serial index.
   * @param var Variable for which to construct the coordinate.
   */
  Coord(const int i, const Var* var);

  /**
   * Increment to next coordinate in serial ordering.
   */
  void inc();

  /**
   * Decrement to previous coordinate in serial ordering.
   */
  void dec();

  /**
   * Recover serial index.
   *
   * @return Serial index for coordinate.
   */
  int index() const;

  /**
   * Prefix increment operator.
   */
  Coord& operator++() {
    inc();
    return *this;
  }

  /**
   * Postfix increment operator.
   */
  Coord operator++(int) {
    Coord tmp(*this);
    inc();
    return tmp;
  }

  /**
   * Prefix decrement operator.
   */
  Coord& operator--() {
    dec();
    return *this;
  }

  /**
   * Postfix decrement operator.
   */
  Coord operator--(int) {
    Coord tmp(*this);
    dec();
    return tmp;
  }

private:
  /**
   * Coordinates.
   */
  std::vector<int> coords;

  /**
   * Sizes along dimensions.
   */
  std::vector<int> sizes;
};
}

#endif
