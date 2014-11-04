/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
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
class Type: public virtual Named, public boost::enable_shared_from_this<Type> {
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
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;
};
}

inline biprog::Type::Type(const char* name) :
    Named(name) {
  //
}

inline biprog::Type::~Type() {
  //
}

inline bool biprog::Type::operator<(const Expression& o) const {
  return false;
}

inline bool biprog::Type::operator<=(const Expression& o) const {
  return operator==(o);
}

inline bool biprog::Type::operator>(const Expression& o) const {
  return false;
}

inline bool biprog::Type::operator>=(const Expression& o) const {
  return operator==(o);
}

inline bool biprog::Type::operator==(const Expression& o) const {
  try {
    const Type& expr = dynamic_cast<const Type&>(o);
    ///@todo Avoid string comparison.
    return name.compare(expr.name) == 0;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Type::operator!=(const Expression& o) const {
  try {
    const Type& expr = dynamic_cast<const Type&>(o);
    ///@todo Avoid string comparison.
    return name.compare(expr.name) != 0;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
