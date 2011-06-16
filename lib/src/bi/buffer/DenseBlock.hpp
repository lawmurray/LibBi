/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_DENSEBLOCK_HPP
#define BI_BUFFER_DENSEBLOCK_HPP

#include "../state/Coord.hpp"
#include "../math/locatable.hpp"
#include "../cuda/cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Represents a dense block across spatial dimensions, giving multiple node
 * ids, where all nodes are dense.
 *
 * @ingroup io_mask
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class DenseBlockBase {
  friend class DenseBlockBase<ON_HOST>;
  friend class DenseBlockBase<ON_DEVICE>;
public:
  /**
   * Vector type.
   */
  typedef typename locatable_temp_vector<L,int>::type vector_type;

  /**
   * Default constructor.
   */
  DenseBlockBase();

  /**
   * Constructor.
   *
   * @param lenX Length of x-dimension.
   * @param lenY Length of y-dimension.
   * @param lenZ Length of z-dimension.
   */
  DenseBlockBase(const int lenX, const int lenY, const int lenZ);

  /**
   * Copy constructor. Ensures deep copy of @p ids.
   */
  DenseBlockBase(const DenseBlockBase<L>& o);

  /**
   * Copy constructor. Ensures deep copy of @p ids.
   */
  template<Location L2>
  DenseBlockBase(const DenseBlockBase<L2>& o);

  /**
   * Assignment operator.
   */
  DenseBlockBase<L>& operator=(const DenseBlockBase<L>& o);

  /**
   * Generic assignment operator.
   */
  template<Location L2>
  DenseBlockBase<L>& operator=(const DenseBlockBase<L2>& o);

  /**
   * Get ids.
   *
   * @return Ids.
   */
  const vector_type& getIds() const;

  /**
   * Get ids.
   *
   * @return Ids.
   */
  vector_type& getIds();

  /**
   * Set id.
   *
   * @param id Id.
   */
  void set(const int id);

  /**
   * Set ids.
   *
   * @tparam V2 Integral vector type.
   *
   * @param ids Ids.
   */
  template<class V2>
  void set(const V2& ids);

  /**
   * Number of active records. This is the product of the number of node ids
   * and the lengths of each dimension.
   */
  int size() const;

  /**
   * Translate dense linear index into variable id and coordinate.
   *
   * @param i dense linear index.
   * @param[out] id Variable id.
   * @param[out] coord Coordinate.
   */
  void coord(const int i, int& id, Coord& coord) const;

  /**
   * Translate dense linear index into variable id.
   *
   * @param i Dense linear index.
   *
   * @return Variable id.
   */
  int id(const int i) const;

  /**
   * Translate dense linear index into dense linear index (identity function).
   *
   * @param i Dense linear index.
   *
   * @return Dense linear index.
   */
  int index(const int i) const;

protected:
  /**
   * Ids of nodes.
   */
  vector_type ids;

  /*
   * Dimension lengths.
   */
  int lenX, lenY, lenZ;
};
}

template<bi::Location L>
inline bi::DenseBlockBase<L>::DenseBlockBase() : lenX(1), lenY(1), lenZ(1) {
  //
}

template<bi::Location L>
inline bi::DenseBlockBase<L>::DenseBlockBase(const int lenX, const int lenY,
    const int lenZ) : lenX(lenX), lenY(lenY), lenZ(lenZ) {
  //
}

template<bi::Location L>
inline bi::DenseBlockBase<L>::DenseBlockBase(const DenseBlockBase<L>& o) :
    ids(o.ids), lenX(o.lenX), lenY(o.lenY), lenZ(o.lenZ) {
  //
}

template<bi::Location L>
template<bi::Location L2>
inline bi::DenseBlockBase<L>::DenseBlockBase(const DenseBlockBase<L2>& o) :
    ids(o.ids), lenX(o.lenX), lenY(o.lenY), lenZ(o.lenZ) {
  //
}

template<bi::Location L>
inline bi::DenseBlockBase<L>& bi::DenseBlockBase<L>::operator=(
    const DenseBlockBase<L>& o) {
  ids.resize(o.ids.size());
  ids = o.ids;
  lenX = o.lenX;
  lenY = o.lenY;
  lenZ = o.lenZ;

  return *this;
}

template<bi::Location L>
template<bi::Location L2>
inline bi::DenseBlockBase<L>& bi::DenseBlockBase<L>::operator=(
    const DenseBlockBase<L2>& o) {
  ids.resize(o.ids.size());
  ids = o.ids;
  lenX = o.lenX;
  lenY = o.lenY;
  lenZ = o.lenZ;

  return *this;
}

template<bi::Location L>
inline const typename bi::DenseBlockBase<L>::vector_type& bi::DenseBlockBase<L>::getIds() const {
  return ids;
}

template<bi::Location L>
inline typename bi::DenseBlockBase<L>::vector_type& bi::DenseBlockBase<L>::getIds() {
  return ids;
}

template<bi::Location L>
inline void bi::DenseBlockBase<L>::set(const int id) {
  this->ids.resize(1);
  *this->ids.begin() = id;
}

template<bi::Location L>
template<class V2>
inline void bi::DenseBlockBase<L>::set(const V2& ids) {
  this->ids.resize(ids.size());
  this->ids = ids;
}

template<bi::Location L>
inline int bi::DenseBlockBase<L>::size() const {
  return ids.size()*lenX*lenY*lenZ;
}

namespace bi {
/**
 * Dense block.
 *
 * @ingroup io_mask
 */
template<Location L = ON_HOST>
class DenseBlock : public DenseBlockBase<L> {
  //
};

/**
 * DenseBlock host specialisation.
 *
 * @ingroup io_mask
 */
template<>
class DenseBlock<ON_HOST> : public DenseBlockBase<ON_HOST> {
public:
  /**
   * @copydoc DenseBlockBase::DenseBlockBase()
   */
  DenseBlock();

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const int, const int, const int)
   */
  DenseBlock(const int lenX, const int lenY, const int lenZ);

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const DenseBlockBase<L>& o)
   */
  DenseBlock(const DenseBlock<ON_HOST>& o);

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const DenseBlockBase<L>& o)
   */
  template<Location L2>
  DenseBlock(const DenseBlock<L2>& o);

  /**
   * @copydoc DenseBlockBase::operator=(const DenseBlockBase<L>& o)
   */
  DenseBlock<ON_HOST>& operator=(const DenseBlock<ON_HOST>& o);

  /**
   * @copydoc DenseBlockBase::operator=(const DenseBlockBase<L2>& o)
   */
  template<Location L2>
  DenseBlock<ON_HOST>& operator=(const DenseBlock<L2>& o);

  /**
   * @copydoc DenseBlockBase::coord()
   */
  void coord(const int i, int& id, Coord& coord) const;

  /**
   * @copydoc DenseBlockBase::id()
   */
  int id(const int i) const;

  /**
   * @copydoc DenseBlockBase::index()
   */
  int index(const int i) const;
};
}

inline bi::DenseBlock<bi::ON_HOST>::DenseBlock() {
  //
}

inline bi::DenseBlock<bi::ON_HOST>::DenseBlock(const int lenX,
    const int lenY, const int lenZ) :
    DenseBlockBase<ON_HOST>(lenX, lenY, lenZ) {
  //
}

inline bi::DenseBlock<bi::ON_HOST>::DenseBlock(
    const DenseBlock<ON_HOST>& o) : DenseBlockBase<ON_HOST>(o) {
  //
}

template<bi::Location L2>
inline bi::DenseBlock<bi::ON_HOST>::DenseBlock(
    const DenseBlock<L2>& o) : DenseBlockBase<ON_HOST>(o) {
  //
}

inline bi::DenseBlock<bi::ON_HOST>& bi::DenseBlock<bi::ON_HOST>::operator=(
    const DenseBlock<ON_HOST>& o) {
  DenseBlockBase<ON_HOST>::operator=(o);
  return *this;
}

template<bi::Location L2>
inline bi::DenseBlock<bi::ON_HOST>& bi::DenseBlock<bi::ON_HOST>::operator=(
    const DenseBlock<L2>& o) {
  DenseBlockBase<ON_HOST>::operator=(o);
  return *this;
}

inline void bi::DenseBlock<bi::ON_HOST>::coord(const int i, int& id,
    Coord& coord) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  int k = i/(lenX*lenY*lenZ);
  int j = i - k*lenX*lenY*lenZ;

  id = *(ids.begin() + k);
  coord.z = j/(lenX*lenY);
  j -= coord.z*lenX*lenY;
  coord.y = j/lenX;
  j -= coord.y*lenX;
  coord.x = j;
}

inline int bi::DenseBlock<bi::ON_HOST>::id(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  int k = i/(lenX*lenY*lenZ);

  return *(ids.begin() + k);
}

inline int bi::DenseBlock<bi::ON_HOST>::index(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  return i;
}

namespace bi {
/**
 * DenseBlock device specialisation.
 *
 * @ingroup io_mask
 */
template<>
class DenseBlock<ON_DEVICE> : public DenseBlockBase<ON_DEVICE> {
public:
  /**
   * @copydoc DenseBlockBase::DenseBlockBase()
   */
  DenseBlock();

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const int, const int, const int)
   */
  DenseBlock(const int lenX, const int lenY, const int lenZ);

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const DenseBlockBase<L>& o)
   */
  DenseBlock(const DenseBlock<ON_DEVICE>& o);

  /**
   * @copydoc DenseBlockBase::DenseBlockBase(const DenseBlockBase<L>& o)
   */
  template<Location L2>
  DenseBlock(const DenseBlock<L2>& o);

  /**
   * @copydoc DenseBlockBase::operator=(const DenseBlockBase<L>& o)
   */
  DenseBlock<ON_DEVICE>& operator=(const DenseBlock<ON_DEVICE>& o);

  /**
   * @copydoc DenseBlockBase::operator=(const DenseBlockBase<L2>& o)
   */
  template<Location L2>
  DenseBlock<ON_DEVICE>& operator=(const DenseBlock<L2>& o);

  /**
   * @copydoc DenseBlockBase::coord()
   */
  CUDA_FUNC_DEVICE void coord(const int i, int& id, Coord& coord) const;

  /**
   * @copydoc DenseBlockBase::id()
   */
  CUDA_FUNC_DEVICE int id(const int i) const;

  /**
   * @copydoc DenseBlockBase::index()
   */
  CUDA_FUNC_DEVICE int index(const int i) const;
};
}

inline bi::DenseBlock<bi::ON_DEVICE>::DenseBlock() {
  //
}

inline bi::DenseBlock<bi::ON_DEVICE>::DenseBlock(const int lenX,
    const int lenY, const int lenZ) :
    DenseBlockBase<ON_DEVICE>(lenX, lenY, lenZ) {
  //
}

inline bi::DenseBlock<bi::ON_DEVICE>::DenseBlock(
    const DenseBlock<bi::ON_DEVICE>& o) : DenseBlockBase<ON_DEVICE>(o) {
  //
}

template<bi::Location L2>
inline bi::DenseBlock<bi::ON_DEVICE>::DenseBlock(
    const DenseBlock<L2>& o) : DenseBlockBase<ON_DEVICE>(o) {
  //
}

inline bi::DenseBlock<bi::ON_DEVICE>& bi::DenseBlock<bi::ON_DEVICE>::operator=(
    const DenseBlock<ON_DEVICE>& o) {
  DenseBlockBase<ON_DEVICE>::operator=(o);
  return *this;
}

template<bi::Location L2>
inline bi::DenseBlock<bi::ON_DEVICE>& bi::DenseBlock<bi::ON_DEVICE>::operator=(
    const DenseBlock<L2>& o) {
  DenseBlockBase<ON_DEVICE>::operator=(o);
  return *this;
}

inline void bi::DenseBlock<bi::ON_DEVICE>::coord(const int i, int& id,
    Coord& coord) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  int k = i/(lenX*lenY*lenZ);
  int j = i - k*lenX*lenY*lenZ;

  id = ids[k];
  coord.z = j/(lenX*lenY);
  j -= coord.z*lenX*lenY;
  coord.y = j/lenX;
  j -= coord.y*lenX;
  coord.x = j;
}

inline int bi::DenseBlock<bi::ON_DEVICE>::id(const int i) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  int k = i/(lenX*lenY*lenZ);

  return ids[k];
}

inline int bi::DenseBlock<bi::ON_DEVICE>::index(const int i) const {
  /* pre-condition */
  //assert (i >= 0 && i < size());

  return i;
}

#endif
