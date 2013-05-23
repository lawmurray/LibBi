/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2867 $
 * $Date: 2012-07-31 15:38:06 +0800 (Tue, 31 Jul 2012) $
 */
#ifndef BI_MISC_SERIALIZATION_HPP
#define BI_MISC_SERIALIZATION_HPP

namespace bi {
/**
 * Load matrix from archive without strict size.
 *
 * @tparam Archive Archive type.
 * @tparam M1 Matrix type.
 *
 * @param ar Archive.
 * @param version Version.
 * @param X Matrix.
 */
template<class Archive, class M1>
void load_resizable_matrix(Archive& ar, const unsigned version, M1& X);

/**
 * Save matrix to archive for restoration without strict size.
 *
 * @tparam Archive Archive type.
 * @tparam M1 Matrix type.
 *
 * @param ar Archive.
 * @param version Version.
 * @param X Matrix.
 */
template<class Archive, class M1>
void save_resizable_matrix(Archive& ar, const unsigned version, const M1& X);

/**
 * Load vector from archive without strict size.
 *
 * @tparam Archive Archive type.
 * @tparam V1 Vector type.
 *
 * @param ar Archive.
 * @param version Version.
 * @param x Vector.
 */
template<class Archive, class V1>
void load_resizable_vector(Archive& ar, const unsigned version, V1& x);

/**
 * Save vector to archive for restoration without strict size.
 *
 * @tparam Archive Archive type.
 * @tparam V1 Vector type.
 *
 * @param ar Archive.
 * @param version Version.
 * @param x Vector.
 */
template<class Archive, class V1>
void save_resizable_vector(Archive& ar, const unsigned version, const V1& x);

}

template<class Archive, class M1>
void bi::load_resizable_matrix(Archive& ar, const unsigned version, M1& X) {
  int rows, cols;
  ar & rows;
  ar & cols;
  X.resize(rows, cols, false);
  ar & X;
}

template<class Archive, class M1>
void bi::save_resizable_matrix(Archive& ar, const unsigned version,
    const M1& X) {
  int rows = X.size1(), cols = X.size2();
  ar & rows;
  ar & cols;
  ar & X;
}

template<class Archive, class V1>
void bi::load_resizable_vector(Archive& ar, const unsigned version, V1& x) {
  int size;
  ar & size;
  x.resize(size, false);
  ar & x;
}

template<class Archive, class V1>
void bi::save_resizable_vector(Archive& ar, const unsigned version,
    const V1& x) {
  int size = x.size();
  ar & size;
  ar & x;
}

#endif
