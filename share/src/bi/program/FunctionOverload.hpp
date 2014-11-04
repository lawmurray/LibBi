/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTIONOVERLOAD_HPP
#define BI_PROGRAM_FUNCTIONOVERLOAD_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * FunctionOverload.
 *
 * @ingroup program
 */
class FunctionOverload: public virtual Named,
    public virtual Parenthesised,
    public virtual Braced,
    public boost::enable_shared_from_this<FunctionOverload> {
public:
  /**
   * Constructor.
   */
  FunctionOverload(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~FunctionOverload();

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

inline biprog::FunctionOverload::FunctionOverload(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::FunctionOverload::~FunctionOverload() {
  //
}

inline bool biprog::FunctionOverload::operator<(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::FunctionOverload::operator<=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::FunctionOverload::operator>(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::FunctionOverload::operator>=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::FunctionOverload::operator==(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::FunctionOverload::operator!=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
