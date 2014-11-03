/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPE_HPP
#define BI_PROGRAM_TYPE_HPP

#include "Named.hpp"

namespace biprog {
/**
 * Type.
 *
 * @ingroup program
 */
class Type: public Named, public boost::enable_shared_from_this<Type> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Type(const char* name);

  /**
   * Destructor.
   */
  virtual ~Type();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Type& o) const;
  virtual bool operator<=(const Type& o) const;
  virtual bool operator>(const Type& o) const;
  virtual bool operator>=(const Type& o) const;
  virtual bool operator==(const Type& o) const;
  virtual bool operator!=(const Type& o) const;
};
}

inline biprog::Type::Type(const char* name) :
    Named(name) {
  //
}

inline biprog::Type::~Type() {
  //
}

inline bool biprog::Type::operator<(const Type& o) const {
  return false;
}

inline bool biprog::Type::operator<=(const Type& o) const {
  return operator==(o);
}

inline bool biprog::Type::operator>(const Type& o) const {
  return false;
}

inline bool biprog::Type::operator>=(const Type& o) const {
  return operator==(o);
}

inline bool biprog::Type::operator==(const Type& o) const {
  ///@todo Avoid string comparison.
  return name.compare(o.name) == 0;
}

inline bool biprog::Type::operator!=(const Type& o) const {
  ///@todo Avoid string comparison.
  return name.compare(o.name) != 0;
}

#endif
