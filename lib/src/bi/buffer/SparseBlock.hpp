/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEBLOCK_HPP
#define BI_BUFFER_SPARSEBLOCK_HPP

#include "../state/Coord.hpp"
#include "../math/locatable.hpp"
#include "../cuda/cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Represents a sparse block across spatial dimensions, giving multiple node
 * ids and multiple coordinates, where all combinations between the two give
 * active variables.
 *
 * @ingroup io_mask
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class SparseBlockBase {
  friend class SparseBlockBase<ON_HOST>;
  friend class SparseBlockBase<ON_DEVICE>;
public:
  /**
   * Vector type.
   */
  typedef typename locatable_temp_vector<L,int>::type vector_type;

  /**
   * Matrix type.
   */
  typedef typename locatable_temp_matrix<L,int>::type matrix_type;

  /**
   * Default constructor.
   */
  SparseBlockBase();

  /**
   * Constructor.
   *
   * @param lenX Length of x-dimension.
   * @param lenY Length of y-dimension.
   * @param lenZ Length of z-dimension.
   */
  SparseBlockBase(const int lenX, const int lenY, const int lenZ);

  /**
   * Copy constructor. Ensures deep copy of @p ids and @p coords.
   */
  SparseBlockBase(const SparseBlockBase<L>& o);

  /**
   * Generic copy constructor. Ensures deep copy of @p ids and @p coords.
   */
  template<Location L2>
  SparseBlockBase(const SparseBlockBase<L2>& o);

  /**
   * Assignment operator.
   */
  SparseBlockBase<L>& operator=(const SparseBlockBase<L>& o);

  /**
   * Generic assignment operator.
   */
  template<Location L2>
  SparseBlockBase<L>& operator=(const SparseBlockBase<L2>& o);

  /**
   * Get ids.
   *
   * @return Ids.
   */
  const vector_type& getIds() const;

  /**
   * Get coordinates.
   */
  const matrix_type& getCoords() const;

  /**
   * Get ids.
   *
   * @return Ids.
   */
  vector_type& getIds();

  /**
   * Get coordinates.
   */
  matrix_type& getCoords();

  /**
   * Set ids and coordinates.
   *
   * @tparam V2 Integral vector type.
   * @tparam M2 Integral matrix type.
   *
   * @param ids Ids.
   * @param coords Coordinates.
   */
  template<class V2, class M2>
  void set(const V2& ids, const M2& coords);

  /**
   * Number of active records. This is the product of the number of node ids
   * with the number of coordinates.
   */
  int size() const;

  /**
   * Translate sparse linear index into variable id and coordinate.
   *
   * @param i Sparse linear index.
   * @param[out] id Variable id.
   * @param[out] coord Coordinate.
   */
  void coord(const int i, int& id, Coord& coord) const;

  /**
   * Translate sparse linear index into variable id.
   *
   * @param i Sparse linear index.
   *
   * @return Variable id.
   */
  int id(const int i) const;

  /**
   * Translate sparse linear index into dense linear index.
   *
   * @param i Sparse linear index.
   *
   * @return Dense linear index.
   */
  int index(const int i) const;

protected:
  /**
   * Ids of nodes.
   */
  vector_type ids;

  /**
   * Coordinates of nodes. Each row matches up to an entry in #ids, columns
   * give ordinates.
   */
  matrix_type coords;

  /**
   * Dimension lengths.
   *
   * @todo Make template parameters?
   */
  int lenX, lenY, lenZ;
};
}

template<bi::Location L>
inline bi::SparseBlockBase<L>::SparseBlockBase() : lenX(1), lenY(1), lenZ(1) {
  //
}

template<bi::Location L>
inline bi::SparseBlockBase<L>::SparseBlockBase(const int lenX, const int lenY,
    const int lenZ) : lenX(lenX), lenY(lenY), lenZ(lenZ) {
  //
}

template<bi::Location L>
inline bi::SparseBlockBase<L>::SparseBlockBase(const SparseBlockBase<L>& o) :
    ids(o.ids), coords(o.coords), lenX(o.lenX), lenY(o.lenY), lenZ(o.lenZ) {
  //
}

template<bi::Location L>
template<bi::Location L2>
inline bi::SparseBlockBase<L>::SparseBlockBase(const SparseBlockBase<L2>& o) :
    ids(o.ids), coords(o.coords), lenX(o.lenX), lenY(o.lenY), lenZ(o.lenZ) {
  //
}

template<bi::Location L>
inline bi::SparseBlockBase<L>& bi::SparseBlockBase<L>::operator=(
    const SparseBlockBase<L>& o) {
  ids.resize(o.ids.size());
  coords.resize(o.coords.size1(), o.coords.size2());
  ids = o.ids;
  coords = o.coords;
  lenX = o.lenX;
  lenY = o.lenY;
  lenZ = o.lenZ;

  return *this;
}

template<bi::Location L>
template<bi::Location L2>
inline bi::SparseBlockBase<L>& bi::SparseBlockBase<L>::operator=(
    const SparseBlockBase<L2>& o) {
  ids.resize(o.ids.size());
  coords.resize(o.coords.size1(), o.coords.size2());
  ids = o.ids;
  coords = o.coords;
  lenX = o.lenX;
  lenY = o.lenY;
  lenZ = o.lenZ;

  return *this;
}

template<bi::Location L>
inline const typename bi::SparseBlockBase<L>::vector_type&
    bi::SparseBlockBase<L>::getIds() const {
  return ids;
}

template<bi::Location L>
inline const typename bi::SparseBlockBase<L>::matrix_type&
    bi::SparseBlockBase<L>::getCoords() const {
  return coords;
}

template<bi::Location L>
inline typename bi::SparseBlockBase<L>::vector_type&
    bi::SparseBlockBase<L>::getIds() {
  return ids;
}

template<bi::Location L>
inline typename bi::SparseBlockBase<L>::matrix_type&
    bi::SparseBlockBase<L>::getCoords() {
  return coords;
}

template<bi::Location L>
template<class V2, class M2>
inline void bi::SparseBlockBase<L>::set(const V2& ids, const M2& coords) {
  this->ids.resize(ids.size());
  this->coords.resize(coords.size1(), coords.size2());
  this->ids = ids;
  this->coords = coords;
}

template<bi::Location L>
inline int bi::SparseBlockBase<L>::size() const {
  return ids.size()*coords.size1();
}

namespace bi {
/**
 * Sparse block.
 */
template<Location L = ON_HOST>
class SparseBlock : public SparseBlockBase<L> {
  //
};

/**
 * SparseBlock host specialisation.
 *
 * @ingroup io_mask
 */
template<>
class SparseBlock<ON_HOST> : public SparseBlockBase<ON_HOST> {
public:
  /**
   * @copydoc SparseBlockBase::SparseBlockBase()
   */
  SparseBlock();

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const int, const int, const int)
   */
  SparseBlock(const int lenX, const int lenY, const int lenZ);

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const SparseBlockBase<L>& o)
   */
  SparseBlock(const SparseBlock<ON_HOST>& o);

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const SparseBlockBase<L2>& o)
   */
  template<Location L2>
  SparseBlock(const SparseBlock<L2>& o);

  /**
   * @copydoc SparseBlockBase::operator=(const SparseBlockBase<L>& o)
   */
  SparseBlock<ON_HOST>& operator=(const SparseBlock<ON_HOST>& o);

  /**
   * @copydoc SparseBlockBase::operator=(const SparseBlockBase<L2>& o)
   */
  template<Location L2>
  SparseBlock<ON_HOST>& operator=(const SparseBlock<L2>& o);

  /**
   * @copydoc SparseBlockBase::coord()
   */
  void coord(const int i, int& id, Coord& coord) const;

  /**
   * @copydoc SparseBlockBase::id()
   */
  int id(const int i) const;

  /**
   * @copydoc SparseBlockBase::index()
   */
  int index(const int i) const;
};
}

inline bi::SparseBlock<bi::ON_HOST>::SparseBlock() {
  //
}

inline bi::SparseBlock<bi::ON_HOST>::SparseBlock(const int lenX,
    const int lenY, const int lenZ) :
    SparseBlockBase<ON_HOST>(lenX, lenY, lenZ) {
  //
}

inline bi::SparseBlock<bi::ON_HOST>::SparseBlock(
    const SparseBlock<ON_HOST>& o) : SparseBlockBase<ON_HOST>(o) {
  //
}

template<bi::Location L2>
inline bi::SparseBlock<bi::ON_HOST>::SparseBlock(
    const SparseBlock<L2>& o) : SparseBlockBase<ON_HOST>(o) {
  //
}

inline bi::SparseBlock<bi::ON_HOST>& bi::SparseBlock<bi::ON_HOST>::operator=(
    const SparseBlock<ON_HOST>& o) {
  SparseBlockBase<ON_HOST>::operator=(o);
  return *this;
}

template<bi::Location L2>
inline bi::SparseBlock<bi::ON_HOST>& bi::SparseBlock<bi::ON_HOST>::operator=(
    const SparseBlock<L2>& o) {
  SparseBlockBase<ON_HOST>::operator=(o);
  return *this;
}

inline void bi::SparseBlock<bi::ON_HOST>::coord(const int i, int& id,
    Coord& coord) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  int k = i/coords.size1();
  int j = i - k*coords.size1();

  id = *(ids.begin() + k);
  coord.x = coords(j,0);
  coord.y = (coords.size2() > 1) ? coords(j,1) : 0;
  coord.z = (coords.size2() > 2) ? coords(j,2) : 0;
}

inline int bi::SparseBlock<bi::ON_HOST>::id(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  int k = i/coords.size1();

  return *(ids.begin() + k);
}

inline int bi::SparseBlock<bi::ON_HOST>::index(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  int id;
  Coord coord;
  this->coord(i, id, coord);

  return coord.z*lenY*lenX + coord.y*lenX + coord.x;
}

namespace bi {
/**
 * SparseBlock device specialisation.
 *
 * @ingroup io_mask
 */
template<>
class SparseBlock<ON_DEVICE> : public SparseBlockBase<ON_DEVICE> {
public:
  /**
   * @copydoc SparseBlockBase::SparseBlockBase()
   */
  SparseBlock();

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const int, const int, const int)
   */
  SparseBlock(const int lenX, const int lenY, const int lenZ);

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const SparseBlockBase<L>& o)
   */
  SparseBlock(const SparseBlock<ON_DEVICE>& o);

  /**
   * @copydoc SparseBlockBase::SparseBlockBase(const SparseBlockBase<L2>& o)
   */
  template<Location L2>
  SparseBlock(const SparseBlock<L2>& o);

  /**
   * @copydoc SparseBlockBase::operator=(const SparseBlockBase<L>& o)
   */
  SparseBlock<ON_DEVICE>& operator=(const SparseBlock<ON_DEVICE>& o);

  /**
   * @copydoc SparseBlockBase::operator=(const SparseBlockBase<L2>& o)
   */
  template<Location L2>
  SparseBlock<ON_DEVICE>& operator=(const SparseBlock<L2>& o);

  /**
   * @copydoc SparseBlockBase::coord()
   */
  CUDA_FUNC_DEVICE void coord(const int i, int& id, Coord& coord) const;

  /**
   * @copydoc SparseBlockBase::id()
   */
  CUDA_FUNC_DEVICE int id(const int i) const;

  /**
   * @copydoc SparseBlockBase::index()
   */
  CUDA_FUNC_DEVICE int index(const int i) const;
};
}

inline bi::SparseBlock<bi::ON_DEVICE>::SparseBlock() {
  //
}

inline bi::SparseBlock<bi::ON_DEVICE>::SparseBlock(const int lenX,
    const int lenY, const int lenZ) :
    SparseBlockBase<ON_DEVICE>(lenX, lenY, lenZ) {
  //
}

inline bi::SparseBlock<bi::ON_DEVICE>::SparseBlock(
    const SparseBlock<bi::ON_DEVICE>& o) {
  operator=(o);
}

template<bi::Location L2>
inline bi::SparseBlock<bi::ON_DEVICE>::SparseBlock(
    const SparseBlock<L2>& o) {
  operator=(o);
}

inline bi::SparseBlock<bi::ON_DEVICE>& bi::SparseBlock<bi::ON_DEVICE>::operator=(
    const SparseBlock<ON_DEVICE>& o) {
  SparseBlockBase<ON_DEVICE>::operator=(o);
  return *this;
}

template<bi::Location L2>
inline bi::SparseBlock<bi::ON_DEVICE>& bi::SparseBlock<bi::ON_DEVICE>::operator=(
    const SparseBlock<L2>& o) {
  SparseBlockBase<ON_DEVICE>::operator=(o);
  return *this;
}

inline void bi::SparseBlock<bi::ON_DEVICE>::coord(const int i, int& id,
    Coord& coord) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  int k = i/coords.size1();
  int j = i - k*coords.size1();

  id = ids[k];
  coord.x = coords(j,0);
  coord.y = (coords.size2() > 1) ? coords(j,1) : 0;
  coord.z = (coords.size2() > 2) ? coords(j,2) : 0;
}

inline int bi::SparseBlock<bi::ON_DEVICE>::id(const int i) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  int k = i/coords.size1();

  return ids[k];
}

inline int bi::SparseBlock<bi::ON_DEVICE>::index(const int i) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  int id;
  Coord coord;
  this->coord(i, id, coord);

  return coord.z*lenY*lenX + coord.y*lenX + coord.x;
}

#endif
