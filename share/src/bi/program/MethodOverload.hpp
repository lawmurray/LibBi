/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHODOVERLOAD_HPP
#define BI_PROGRAM_METHODOVERLOAD_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * MethodOverload.
 *
 * @ingroup program
 */
class MethodOverload: public virtual Named,
    public virtual Parenthesised,
    public virtual Braced,
    public virtual Scoped,
    public boost::enable_shared_from_this<MethodOverload> {
public:
  /**
   * Constructor.
   */
  MethodOverload(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~MethodOverload();

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

inline biprog::MethodOverload::MethodOverload(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Braced(braces), Scoped(scope) {
  //
}

inline biprog::MethodOverload::~MethodOverload() {
//
}

inline bool biprog::MethodOverload::operator<(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::MethodOverload::operator<=(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::MethodOverload::operator>(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::MethodOverload::operator>=(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::MethodOverload::operator==(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::MethodOverload::operator!=(const Expression& o) const {
  try {
    const MethodOverload& expr = dynamic_cast<const MethodOverload&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
