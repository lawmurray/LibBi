/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_COORD_HPP
#define BI_STATE_COORD_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Coordinate of node.
 *
 * @ingroup state
 */
struct Coord {
  /**
   * Default constructor.
   *
   * Members are not initialised.
   */
  Coord();

  /**
   * Constructor.
   */
  Coord(const int x, const int y, const int z);

  /**
   * Factory method for translating node index to coordinate, where node
   * type is known.
   *
   * @tparam B Model type.
   * @tparam X Node type.
   *
   * @param i Offset into buffer.
   */
  template<class B, class X>
  static CUDA_FUNC_BOTH Coord make(const int i);

  /**
   * Calculate index into buffer for current coordinate and given offset.
   *
   * @tparam B Model type.
   * @tparam X Node type.
   * @tparam Xo X-offset.
   * @tparam Yo Y-offset.
   * @tparam Zo Z-offset.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  CUDA_FUNC_BOTH int index() const;

  /**
   * X-ordinate.
   */
  int x;

  /**
   * Y-ordinate.
   */
  int y;

  /**
   * Z-ordinate.
   */
  int z;
};

}

#include "../model/model.hpp"
#include "../traits/boundary_traits.hpp"

inline bi::Coord::Coord() : x(0), y(0), z(0) {
  //
}

inline bi::Coord::Coord(const int x, const int y,
    const int z) : x(x), y(y), z(z) {
  //
}

template<class B, class X>
inline bi::Coord bi::Coord::make(const int i) {
  const int start = node_start<B,X>::value;
  const int j = i - start; // offset from start of nodes of this type
  int x, y, z;

  z = node_has_z<X>::value ? j / (B::NX*B::NY) : 0;
  y = node_has_y<X>::value ? (j - z*(B::NX*B::NY)) / B::NX : 0;
  x = node_has_x<X>::value ? (j - y*B::NY) : 0;

  return Coord(x, y, z);
}

template<class B, class X, int Xo, int Yo, int Zo>
inline int bi::Coord::index() const {
  int x, y, z;

  if (node_has_x<X>::value) {
    if (has_cyclic_x_boundary<X>::value) {
      x = (this->x + B::NX + Xo) % B::NX;
    } else {
      x = this->x + Xo;
    }
  } else {
    x = 0;
  }

  if (node_has_y<X>::value) {
    if (has_cyclic_y_boundary<X>::value) {
      y = (this->y + B::NY + Yo) % B::NY;
    } else {
      y = this->y + Yo;
    }
  } else {
    y = 0;
  }

  if (node_has_z<X>::value) {
    if (has_cyclic_z_boundary<X>::value) {
      z = (this->z + B::NZ + Zo) % B::NZ;
    } else {
      z = this->z + Zo;
    }
  } else {
    z = 0;
  }

  return node_start<B,X>::value + z*B::NX*B::NY + y*B::NX + x;
}

#endif
